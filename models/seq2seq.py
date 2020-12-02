# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import data.dict as dict
import models
# from figure_hot import relitu_line

import numpy as np


class seq2seq(nn.Module):

    def __init__(self, config, input_emb_size, mix_speech_len, tgt_vocab_size, tgt_dir_vocab_size, use_cuda, pretrain=None, score_fn=''):
        super(seq2seq, self).__init__()
        if pretrain is not None:
            src_embedding = pretrain['src_emb']
            tgt_embedding = pretrain['tgt_emb']
        else:
            src_embedding = None
            tgt_embedding = None
        
        self.encoder = models.rnn_encoder(config, input_emb_size)
        ## TasNet Encoder
        self.TasNetEncoder = models.TasNetEncoder()
        self.TasNetDecoder = models.TasNetDecoder()

        if config.shared_vocab == False:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, tgt_dir_vocab_size, embedding=tgt_embedding, score_fn=score_fn)
        else:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, tgt_dir_vocab_size, embedding=self.encoder.embedding, score_fn=score_fn)
        self.use_cuda = use_cuda
        self.tgt_vocab_size = tgt_vocab_size
        self.tgt_dir_vocab_size = tgt_dir_vocab_size
        self.config = config
        self.criterion = models.criterion(tgt_vocab_size, use_cuda, config.loss)
        self.criterion_dir = models.criterion_dir(tgt_dir_vocab_size, use_cuda, config.loss)
        self.loss_for_ss = nn.MSELoss()
        self.loss_for_dir = nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.linear_output = nn.Linear(tgt_dir_vocab_size, 1)
        self.wav_loss = models.WaveLoss(dBscale=1, nfft=config.FRAME_LENGTH, hop_size=config.FRAME_SHIFT)

        speech_fre = input_emb_size
        num_labels = tgt_vocab_size
        if config.use_tas:
            self.ss_model = models.ConvTasNet()
        else:
            self.ss_model = models.SS(config, speech_fre, mix_speech_len, num_labels)

    def compute_loss(self, hidden_outputs, hidden_outputs_dir, targets, targets_dir, memory_efficiency):
        if memory_efficiency:
            return models.memory_efficiency_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
        else:
            sgm_loss_speaker, num_total, num_correct = models.cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
            #sgm_loss_direction = models.mmse_loss2(hidden_outputs_dir, self.decoder, targets_dir, self.loss_for_dir)
            sgm_loss_direction, num_total_dir, num_correct_dir = models.cross_entropy_loss_dir(hidden_outputs_dir, self.decoder, targets_dir, self.criterion_dir, self.config)
            print("sgm_loss_speaker:", sgm_loss_speaker)
            print("sgm_loss_direction:", sgm_loss_direction)
            sgm_loss = sgm_loss_speaker + sgm_loss_direction
            return sgm_loss, num_total, num_correct, num_total_dir, num_correct_dir

    def separation_loss(self, x_input_map_multi, masks, y_multi_map, Var='NoItem'):
        if not self.config.MLMSE:
            return models.ss_loss(self.config, x_input_map_multi, masks, y_multi_map, self.loss_for_ss,self.wav_loss)
        else:
            return models.ss_loss_MLMSE(self.config, x_input_map_multi, masks, y_multi_map, self.loss_for_ss, Var)

    def separation_tas_loss(self,predict_wav, y_multi_wav,mix_lengths):
        return models.ss_tas_loss(self.config, predict_wav, y_multi_wav, mix_lengths,self.loss_for_ss)

    def update_var(self, x_input_map_multi, multi_masks, y_multi_map):
        predict_multi_map = torch.mean(multi_masks * x_input_map_multi, -2)  # 在时间维度上平均
        y_multi_map = torch.mean(Variable(y_multi_map), -2)  # 在时间维度上平均
        loss_vector = (y_multi_map - predict_multi_map).view(-1, self.config.speech_fre).unsqueeze(-1)  # 应该是bs*1*fre
        Var = torch.bmm(loss_vector, loss_vector.transpose(1, 2))
        Var = torch.mean(Var, 0)  # 在batch的维度上平均
        return Var.detach()

    def forward(self, src, tgt, tgt_dir):
        #lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        # src = torch.index_select(src, dim=0, index=indices)
        # tgt = torch.index_select(tgt, dim=0, index=indices)

        mix_wav = src.transpose(0,1)        # [batch, sample, channel]
        mix = self.TasNetEncoder(mix_wav)   # [batch, featuremap, timeStep]
        mix_infer = mix.transpose(1,2)      # [batch, timeStep, featuremap]
        _, lengths, _ = mix_infer.size()
        # 4 equals to the number of GPU
        lengths = Variable(torch.LongTensor(self.config.batch_size/4).zero_() + lengths).unsqueeze(0).cuda()
        lengths, indices = torch.sort(lengths.squeeze(0), dim=0, descending=True)

        contexts, state = self.encoder(mix_infer, lengths.data.tolist())  # context [max_len,batch_size,hidden_size×2]
        outputs, outputs_dir, final_state, global_embs = self.decoder(tgt[:-1], tgt_dir[:-1], state, contexts.transpose(0, 1))

        if self.config.use_tas:
            predicted_maps = self.ss_model(mix, global_embs.transpose(0,1))

        predicted_signal = self.TasNetDecoder(mix, predicted_maps) # [batch, spkN, timeStep]

        return outputs, outputs_dir, tgt[1:], tgt_dir[1:], predicted_signal.transpose(0,1)

    def sample(self, src, src_len):
        # src=src.squeeze()
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        bos = Variable(torch.ones(src.size(1)).long().fill_(dict.BOS), volatile=True)

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.tolist())
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts.transpose(0, 1))
        _, attns_weight = final_outputs
        alignments = attns_weight.max(2)[1]
        sample_ids = torch.index_select(sample_ids.data, dim=1, index=ind)
        alignments = torch.index_select(alignments.data, dim=1, index=ind)
        # targets = tgt[1:]

        return sample_ids.t(), alignments.t()

    def beam_sample(self, src, dict_spk2idx, dict_dir2idx, beam_size=1):

        mix_wav = src.transpose(0,1)      # [batch, sample]
        mix = self.TasNetEncoder(mix_wav)      # [batch, featuremap, timeStep]
        
        mix_infer = mix.transpose(1,2)    # [batch, timeStep, featuremap]
        batch_size, lengths, _ = mix_infer.size()
        lengths = Variable(torch.LongTensor(self.config.batch_size).zero_() + lengths).unsqueeze(0).cuda()
        lengths, indices = torch.sort(lengths.squeeze(0), dim=0, descending=True)
        
        contexts, encState = self.encoder(mix_infer, lengths.data.tolist())  # context [max_len,batch_size,hidden_size×2]

        #  (1b) Initialize for the decoder.
        def var(a):
            return Variable(a, volatile=True)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = rvar(contexts.data).transpose(0, 1)
        decState = (rvar(encState[0].data), rvar(encState[1].data))
        decState_dir = (rvar(encState[0].data), rvar(encState[1].data))
        # decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, dict_spk2idx, n_best=1,
                            cuda=self.use_cuda) for __ in range(batch_size)]

        beam_dir = [models.Beam(beam_size, dict_dir2idx, n_best=1,
                            cuda=self.use_cuda) for __ in range(batch_size)]
        # (2) run the decoder to generate sentences, using beam search.

        mask = None
        mask_dir = None
        soft_score = None
        tmp_hiddens = []
        tmp_soft_score = []

        soft_score_dir = None
        tmp_hiddens_dir = []
        tmp_soft_score_dir = []
        output_list = []
        output_dir_list = []
        predicted_list = []
        predicted_dir_list = []  
        output_bk_list = []
        output_bk_dir_list = []
        hidden_list = []
        hidden_dir_list = []
        emb_list = []
        emb_dir_list = [] 
        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break
            if all((b_dir.done() for b_dir in beam_dir)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(-1))
            inp_dir = var(torch.stack([b_dir.getCurrentState() for b_dir in beam_dir]).t().contiguous().view(-1))

            # Run one step.
            output, output_dir, decState, decState_dir, attn_weights, attn_weights_dir, hidden, hidden_dir, emb, emb_dir, output_bk, output_bk_dir = self.decoder.sample_one(inp, inp_dir, soft_score, soft_score_dir, decState, decState_dir, tmp_hiddens, tmp_hiddens_dir,
                                                                          contexts, mask, mask_dir)
            soft_score = F.softmax(output)
            soft_score_dir = F.softmax(output_dir)
            
            predicted = output.max(1)[1]
            predicted_dir = output_dir.max(1)[1]
            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                    mask_dir = predicted_dir.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)
                    mask_dir = torch.cat((mask_dir, predicted_dir.unsqueeze(1)), 1)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.

            output_list.append(output[0])
            output_dir_list.append(output_dir[0])

            output = unbottle(self.log_softmax(output))
            output_dir = unbottle(F.sigmoid(output_dir))
            
            attn = unbottle(attn_weights)
            hidden = unbottle(hidden)
            emb = unbottle(emb)
            attn_dir = unbottle(attn_weights_dir)
            hidden_dir = unbottle(hidden_dir)
            emb_dir = unbottle(emb_dir)
            # beam x tgt_vocab

            output_bk_list.append(output_bk[0])
            output_bk_dir_list.append(output_bk_dir[0])
            hidden_list.append(hidden[0])
            hidden_dir_list.append(hidden_dir[0])
            emb_list.append(emb[0])
            emb_dir_list.append(emb_dir[0])

            predicted_list.append(predicted)
            predicted_dir_list.append(predicted_dir)

            # (c) Advance each beam.
            # update state

            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j], hidden.data[:, j], emb.data[:, j])
                b.beam_update(decState, j)  # 这个函数更新了原来的decState,只不过不是用return，是直接赋值！
                if self.config.ct_recu:
                    b.beam_update_context(contexts, j)  # 这个函数更新了原来的decState,只不过不是用return，是直接赋值！
            for i, a in enumerate(beam_dir):
                a.advance(output_dir.data[:, i], attn_dir.data[:, i], hidden_dir.data[:, i], emb_dir.data[:, i])
                a.beam_update(decState_dir, i)  # 这个函数更新了原来的decState,只不过不是用return，是直接赋值！
                if self.config.ct_recu:
                    a.beam_update_context(contexts, i)  # 这个函数更新了原来的decState,只不过不是用return，是直接赋值！
            # print "beam after decState:",decState[0].data.cpu().numpy().mean()

        # (3) Package everything up.
        allHyps,allHyps_dir, allScores, allAttn, allHiddens, allEmbs = [],[], [], [], [], []

        ind = range(batch_size)
        for j in ind:
            b = beam[j]
            c = beam_dir[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, hyps_dir, attn, hiddens, embs = [], [], [], [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att, hidden, emb = b.getHyp(times, k)
                hyp_dir, att_dir, hidden_dir, emb_dir = c.getHyp(times, k)
                if self.config.relitu:
                    relitu_line(626, 1, att[0].cpu().numpy())
                    relitu_line(626, 1, att[1].cpu().numpy())
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                hiddens.append(hidden+hidden_dir)
                embs.append(emb+emb_dir)
                hyps_dir.append(hyp_dir)
            allHyps.append(hyps[0])
            allHyps_dir.append(hyps_dir[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            allHiddens.append(hiddens[0])
            allEmbs.append(embs[0])

        ss_embs = Variable(torch.stack(allEmbs, 0).transpose(0, 1))  # to [decLen, bs, dim]
        if self.config.use_tas:
            predicted_maps = self.ss_model(mix, ss_embs[1:].transpose(0,1))

        predicted_signal = self.TasNetDecoder(mix, predicted_maps) # [batch, spkN, timeStep]
        return allHyps, allHyps_dir, allAttn, allHiddens, predicted_signal.transpose(0,1), output_list, output_dir_list, output_bk_list, output_bk_dir_list, hidden_list, hidden_dir_list, emb_list, emb_dir_list
