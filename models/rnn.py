# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict
import models

import numpy as np


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)  # 把多层的LSTMCell模型的输出给组织起来了，的到了[num_layers,batch_size,hidden_size]的东西
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class rnn_encoder(nn.Module):

    def __init__(self, config, input_emb_size):
        super(rnn_encoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirec)
        self.config = config

    def forward(self, input, lengths):
        input = input.transpose(0, 1)
        embs = pack(input, list(map(int, lengths)))  # 这里batch是第二个维度
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if not self.config.bidirec:
            return outputs, (h, c)  # h,c是最后一个step的，大小是(num_layers * num_directions, batch, hidden_size)
        else:
            batch_size = h.size(1)
            h = h.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            c = c.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            state = (h.transpose(0, 1), c.transpose(0, 1))  # 每一个元素是 (num_layers,batch,2*hidden_size)这么大。
            return outputs, state

class gated_rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(gated_rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        self.gated = nn.Sequential(nn.Linear(config.encoder_hidden_size, 1), nn.Sigmoid())

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        p = self.gated(outputs)
        outputs = outputs * p
        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, dir_vocab_size, embedding=None, score_fn=None):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
            self.embedding_dir = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
            self.embedding_dir = nn.Embedding(dir_vocab_size, config.emb_size)
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                               num_layers=config.num_layers, dropout=config.dropout)
        self.rnn_dir = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                               num_layers=config.num_layers, dropout=config.dropout)

        self.score_fn = score_fn
        if self.score_fn.startswith('general'):
            self.linear = nn.Linear(config.decoder_hidden_size, config.emb_size)
            self.linear_dir = nn.Linear(config.decoder_hidden_size, config.emb_size)
        elif score_fn.startswith('concat'):
            self.linear_query = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
            self.linear_weight = nn.Linear(config.emb_size, config.decoder_hidden_size)
            self.linear_v = nn.Linear(config.decoder_hidden_size, 1)
            self.linear_query_dir = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
            self.linear_weight_dir = nn.Linear(config.emb_size, config.decoder_hidden_size)
            self.linear_v_dir = nn.Linear(config.decoder_hidden_size, 1)
        elif not self.score_fn.startswith('dot'):
            self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)
            self.linear_dir = nn.Linear(config.decoder_hidden_size, dir_vocab_size)
            self.linear_output = nn.Linear(dir_vocab_size, 1)

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None
        
        self.attention = models.global_attention(config.decoder_hidden_size, activation)
        self.attention_dir = models.global_attention(config.decoder_hidden_size, activation)
        self.hidden_size = config.decoder_hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

        if self.config.global_emb:
            self.gated1 = nn.Linear(config.emb_size, config.emb_size)
            self.gated2 = nn.Linear(config.emb_size, config.emb_size)
            self.gated1_dir = nn.Linear(config.emb_size, config.emb_size)
            self.gated2_dir = nn.Linear(config.emb_size, config.emb_size)

    def forward(self, inputs, inputs_dir, init_state, contexts):

        outputs, outputs_dir, state, attns, global_embs = [], [], init_state, [], []
      
        ## speaker
        embs = self.embedding(inputs).split(1)  # time_step [1,bs,embsize]
        max_time_step = len(embs)
        emb = embs[0]  # 第一步BOS的embedding.
        output, state_speaker = self.rnn(emb.squeeze(0), state)
        output, attn_weights = self.attention(output, contexts)
        output = self.dropout(output)
        soft_score = F.softmax(self.linear(output))  # 第一步的概率分布也就是 bs,vocal这么大

        ## direction
        embs_dir = self.embedding_dir(inputs_dir).split(1)  # time_step [1,bs,embsize]
        emb_dir = embs_dir[0]  # 第一步BOS的embedding.
        output_dir, state_dir = self.rnn_dir(emb_dir.squeeze(0), state)
        output_dir, attn_weights_dir = self.attention_dir(output_dir, contexts)
        output_dir = self.dropout(output_dir)
        #soft_score_dir_1 = F.sigmoid(self.linear_dir(output_dir))
        soft_score_dir = F.softmax(self.linear_dir(output_dir))
        
        attn_weights = attn_weights + attn_weights_dir
        outputs += [output]
        outputs_dir += [output_dir]
        attns += [attn_weights]

        batch_size = soft_score.size(0)
        a, b = self.embedding.weight.size()
        c, d = self.embedding_dir.weight.size()

        for i in range(max_time_step - 1):
            ## speaker
            emb1 = torch.bmm(soft_score.unsqueeze(1),
                             self.embedding.weight.expand((batch_size, a, b)))
            emb2 = embs[i + 1]  
            gamma = F.sigmoid(self.gated1(emb1.squeeze(1)) + self.gated2(emb2.squeeze(0))) 
            emb = gamma * emb1.squeeze(1) + (1 - gamma) * emb2.squeeze(0)
            output, state_speaker = self.rnn(emb, state_speaker)
            output, attn_weights = self.attention(output, contexts)
            output = self.dropout(output)
            soft_score = F.softmax(self.linear(output))

            ## direction
            emb1_dir = torch.bmm(soft_score_dir.unsqueeze(1),
                             self.embedding_dir.weight.expand((batch_size, c, d)))
            emb2_dir = embs_dir[i + 1] 
            gamma_dir = F.sigmoid(self.gated1_dir(emb1_dir.squeeze(1)) + self.gated2_dir(emb2_dir.squeeze(0)))
            emb_dir = gamma_dir * emb1_dir.squeeze(1) + (1 - gamma_dir) * emb2_dir.squeeze(0)
            output_dir, state_dir = self.rnn_dir(emb_dir, state_dir)
            output_dir, attn_weights_dir = self.attention_dir(output_dir, contexts)
            output_dir = self.dropout(output_dir)
            #soft_score_dir_1 = F.sigmoid(self.linear_dir(output_dir))
            soft_score_dir = F.softmax(self.linear_dir(output_dir))
            
            attn_weights = attn_weights + attn_weights_dir
            emb = emb + emb_dir
            outputs += [output]
            outputs_dir += [output_dir]
            global_embs += [emb]
            attns += [attn_weights]

        outputs = torch.stack(outputs)
        outputs_dir = torch.stack(outputs_dir)
        global_embs = torch.stack(global_embs)
        attns = torch.stack(attns)
        return outputs, outputs_dir, state, global_embs

    def compute_score(self, hiddens):
        if self.score_fn.startswith('general'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(self.linear(hiddens), Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(self.linear(hiddens), self.embedding.weight.t())
        elif self.score_fn.startswith('concat'):
            if self.score_fn.endswith('not'):
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                                  self.linear_weight(Variable(self.embedding.weight.data)).unsqueeze(
                                                      0))).squeeze(2)
            else:
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                                  self.linear_weight(self.embedding.weight).unsqueeze(0))).squeeze(2)
        elif self.score_fn.startswith('dot'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(hiddens, Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(hiddens, self.embedding.weight.t())
#        elif self.score_fn.startswith('arc_margin'):
#            scores = self.linear(hiddens,targets)
        else:
            scores = self.linear(hiddens)
        return scores

    def compute_score_dir(self, hiddens):
        if self.score_fn.startswith('general'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(self.linear_dir(hiddens), Variable(self.embedding_dir.weight.t().data))
            else:
                scores = torch.matmul(self.linear_dir(hiddens), self.embedding_dir.weight.t())
        elif self.score_fn.startswith('concat'):
            if self.score_fn.endswith('not'):
                scores = self.linear_v_dir(torch.tanh(self.linear_query_dir(hiddens).unsqueeze(1) + \
                                                  self.linear_weight_dir(Variable(self.embedding_dir.weight.data)).unsqueeze(
                                                      0))).squeeze(2)
            else:
                scores = self.linear_v_dir(torch.tanh(self.linear_query_dir(hiddens).unsqueeze(1) + \
                                                  self.linear_weight_dir(self.embedding_dir.weight).unsqueeze(0))).squeeze(2)
        elif self.score_fn.startswith('dot'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(hiddens, Variable(self.embedding_dir.weight.t().data))
            else:
                scores = torch.matmul(hiddens, self.embedding_dir.weight.t())
        else:
            #scores_1 = F.sigmoid(self.linear_dir(hiddens))
            scores = self.linear_dir(hiddens)
        return scores

    def sample(self, input, init_state, contexts):
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len
        soft_score = None
        mask = None
        for i in range(max_time_step):
            output, state, attn_weights = self.sample_one(inputs[i], soft_score, state, contexts, mask)
            if self.config.global_emb:
                soft_score = F.softmax(output)
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]
            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)
        return sample_ids, (outputs, attns)

    def sample_one(self, input, input_dir, soft_score, soft_score_dir, state, state_dir, tmp_hiddens, tmp_hiddens_dir, contexts, mask,mask_dir):
        if self.config.global_emb:
            batch_size = contexts.size(0)
            a, b = self.embedding.weight.size()
            if soft_score is None:
                emb = self.embedding(input)
            else:
                emb1 = torch.bmm(soft_score.unsqueeze(1), self.embedding.weight.expand((batch_size, a, b)))
                emb2 = self.embedding(input)
                gamma = F.sigmoid(self.gated1(emb1.squeeze()) + self.gated2(emb2.squeeze()))
                emb = gamma * emb1.squeeze() + (1 - gamma) * emb2.squeeze()

            c, d = self.embedding_dir.weight.size()
            if soft_score_dir is None:
                emb_dir = self.embedding_dir(input_dir)
            else:
                emb1_dir = torch.bmm(soft_score_dir.unsqueeze(1), self.embedding_dir.weight.expand((batch_size, c, d)))
                emb2_dir = self.embedding_dir(input_dir)
                gamma_dir = F.sigmoid(self.gated1_dir(emb1_dir.squeeze()) + self.gated2_dir(emb2_dir.squeeze()))
                emb_dir = gamma_dir * emb1_dir.squeeze() + (1 - gamma_dir) * emb2_dir.squeeze()
        else:
            emb = self.embedding(input)
            emb_dir = self.embedding_dir(input_dir)

        output, state = self.rnn(emb, state)
        output_bk = output
        hidden, attn_weights = self.attention(output, contexts)
        if self.config.schmidt:
            hidden = models.schmidt(hidden, tmp_hiddens)
        output = self.compute_score(hidden)
        if self.config.mask:
            if mask is not None:
                output = output.scatter_(1, mask, -9999999999)

        output_dir, state_dir = self.rnn_dir(emb_dir, state_dir)
        output_dir_bk = output_dir
        hidden_dir, attn_weights_dir = self.attention_dir(output_dir, contexts)
        if self.config.schmidt:
            hidden_dir = models.schmidt(hidden_dir, tmp_hiddens_dir)
        output_dir = self.compute_score_dir(hidden_dir)
        if self.config.mask:
            if mask_dir is not None:
                output_dir = output_dir.scatter_(1, mask_dir, -9999999999)

        return output, output_dir, state, state_dir, attn_weights, attn_weights_dir, hidden, hidden_dir, emb, emb_dir, output_bk, output_dir_bk
