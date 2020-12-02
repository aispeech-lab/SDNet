# coding=utf8
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

np.random.seed(1)  # 设定种子
torch.manual_seed(1)
random.seed(1)
torch.cuda.set_device(0)
test_all_outputchannel = 0


class ATTENTION(nn.Module):
    def __init__(self, hidden_size, query_size, align_hidden_size, mode='dot'):
        super(ATTENTION, self).__init__()
        # self.mix_emb_size=config.EMBEDDING_SIZE
        self.hidden_size = hidden_size
        self.query_size = query_size
        # self.align_hidden_size=hidden_size #align模式下的隐层大小，暂时取跟原来一致的
        self.align_hidden_size = align_hidden_size  # align模式下的隐层大小，暂时取跟原来一致的
        self.mode = mode
        self.Linear_1 = nn.Linear(self.hidden_size, self.align_hidden_size, bias=False)
        # self.Linear_2=nn.Linear(hidden_sizedw,self.align_hidden_size,bias=False)
        self.Linear_2 = nn.Linear(self.query_size, self.align_hidden_size, bias=False)
        self.Linear_3 = nn.Linear(self.align_hidden_size, 1, bias=False)

    def forward(self, mix_hidden, query):
        # todo:这个要弄好，其实也可以直接抛弃memory来进行attention | DONE
        BATCH_SIZE = mix_hidden.size()[0]
        assert query.size() == (BATCH_SIZE, self.query_size)
        assert mix_hidden.size()[-1] == self.hidden_size
        # mix_hidden：bs,max_len,fre,hidden_size  query:bs,hidden_size
        if self.mode == 'dot':
            # mix_hidden=mix_hidden.view(-1,1,self.hidden_size)
            mix_shape = mix_hidden.size()
            mix_hidden = mix_hidden.view(BATCH_SIZE, -1, self.hidden_size)
            query = query.view(-1, self.hidden_size, 1)
            # print '\n\n',mix_hidden.requires_grad,query.requires_grad,'\n\n'
            dot = torch.baddbmm(Variable(torch.zeros(1, 1)), mix_hidden, query)
            energy = dot.view(BATCH_SIZE, mix_shape[1], mix_shape[2])
            # TODO: 这里可以想想是不是能换成Relu之类的
            mask = F.sigmoid(energy)
            return mask

        elif self.mode == 'align':
            # mix_hidden=Variable(mix_hidden)
            # query=Variable(query)
            mix_shape = mix_hidden.size()
            mix_hidden = mix_hidden.view(-1, self.hidden_size)
            mix_hidden = self.Linear_1(mix_hidden).view(BATCH_SIZE, -1, self.align_hidden_size)
            query = self.Linear_2(query).view(-1, 1, self.align_hidden_size)  # bs,1,hidden
            sum = F.tanh(mix_hidden + query)
            # TODO:从这里开始做起
            energy = self.Linear_3(sum.view(-1, self.align_hidden_size)).view(BATCH_SIZE, mix_shape[1], mix_shape[2])
            mask = F.sigmoid(energy)
            return mask
        else:
            print
            'NO this attention methods.'
            raise IndexError


class MIX_SPEECH_CNN(nn.Module):
    def __init__(self, config, input_fre, mix_speech_len):
        super(MIX_SPEECH_CNN, self).__init__()
        self.input_fre = input_fre
        self.mix_speech_len = mix_speech_len
        self.config = config

        self.cnn1 = nn.Conv2d(1, 96, (1, 7), stride=1, padding=(0, 3), dilation=(1, 1))
        self.cnn2 = nn.Conv2d(96, 96, (7, 1), stride=1, padding=(3, 0), dilation=(1, 1))
        self.cnn3 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(2, 2), dilation=(1, 1))
        self.cnn4 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(4, 2), dilation=(2, 1))
        self.cnn5 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(8, 2), dilation=(4, 1))

        self.cnn6 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(16, 2), dilation=(8, 1))
        self.cnn7 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(32, 2), dilation=(16, 1))
        self.cnn8 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(64, 2), dilation=(32, 1))
        self.cnn9 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(2, 2), dilation=(1, 1))
        self.cnn10 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(4, 4), dilation=(2, 2))

        self.cnn11 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(8, 8), dilation=(4, 4))
        self.cnn12 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(16, 16), dilation=(8, 8))
        self.cnn13 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(32, 32), dilation=(16, 16))
        self.cnn14 = nn.Conv2d(96, 96, (5, 5), stride=1, padding=(64, 64), dilation=(32, 32))
        self.cnn15 = nn.Conv2d(96, 8, (1, 1), stride=1, padding=(0, 0), dilation=(1, 1))
        self.num_cnns = 15
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(96)
        self.bn5 = nn.BatchNorm2d(96)
        self.bn6 = nn.BatchNorm2d(96)
        self.bn7 = nn.BatchNorm2d(96)
        self.bn8 = nn.BatchNorm2d(96)
        self.bn9 = nn.BatchNorm2d(96)
        self.bn10 = nn.BatchNorm2d(96)
        self.bn11 = nn.BatchNorm2d(96)
        self.bn12 = nn.BatchNorm2d(96)
        self.bn13 = nn.BatchNorm2d(96)
        self.bn14 = nn.BatchNorm2d(96)
        self.bn15 = nn.BatchNorm2d(8)

    def forward(self, x):
        print
        'speech input size:', x.size()
        assert len(x.size()) == 3
        x = x.unsqueeze(1)
        print
        '\nSpeech layer log:'
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer = eval('self.cnn{}'.format(idx + 1))
            bn_layer = eval('self.bn{}'.format(idx + 1))
            x = F.relu(cnn_layer(x))
            x = bn_layer(x)
            print
            'speech shape after CNNs:', idx, '', x.size()

        out = x.transpose(1, 3).transpose(1, 2).contiguous()
        print
        'speech output size:', out.size()
        return out, out


class MIX_SPEECH(nn.Module):
    def __init__(self, config, input_fre, mix_speech_len):
        super(MIX_SPEECH, self).__init__()
        self.input_fre = input_fre
        self.mix_speech_len = mix_speech_len
        self.layer = nn.LSTM(
            input_size=input_fre,
            hidden_size=config.HIDDEN_UNITS,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )
        # self.batchnorm = nn.BatchNorm1d(self.input_fre*config.EMBEDDING_SIZE)
        self.Linear = nn.Linear(2 * config.HIDDEN_UNITS, self.input_fre * config.EMBEDDING_SIZE,bias=1)
        self.config = config

    def forward(self, x):
        x, hidden = self.layer(x)
        batch_size = x.size()[0]
        x = x.contiguous()
        xx = x
        x = x.view(batch_size * self.mix_speech_len, -1)
        # out=F.tanh(self.Linear(x))
        out = self.Linear(x)
        # out = self.batchnorm(out)
        out = F.tanh(out)
        # out=F.relu(out)
        out = out.view(batch_size, self.mix_speech_len, self.input_fre, -1)
        # print 'Mix speech output shape:',out.size()
        return out, xx


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn = nn.Conv2d(1, 64, (3, 3), stride=(2, 2), )
        self.cnn1 = nn.Conv2d(64, 64, (3, 3), stride=(2, 2), )
        self.cnn2 = nn.Conv2d(64, 64, (3, 3), stride=(2, 2), )
        # self.final=nn.Linear(36480,1)
        self.final = nn.Linear(73920, 1)

    def forward(self, spec):
        bs, topk, len, fre = spec.size()
        spec = spec.view(bs * topk, 1, len, fre)
        spec = F.relu(self.cnn(spec))
        spec = F.relu(self.cnn1(spec))
        spec = F.relu(self.cnn2(spec))
        spec = spec.view(bs * topk, -1)
        print
        'size spec:', spec.size()
        score = F.sigmoid(self.final(spec))
        print
        'size spec:', score.size()
        return score


class SPEECH_EMBEDDING(nn.Module):
    def __init__(self, num_labels, embedding_size, max_num_channel):
        super(SPEECH_EMBEDDING, self).__init__()
        self.num_all = num_labels
        self.emb_size = embedding_size
        self.max_num_out = max_num_channel
        # self.layer=nn.Embedding(num_labels,embedding_size,padding_idx=-1)
        self.layer = nn.Embedding(num_labels, embedding_size)

    def forward(self, input, mask_idx):
        aim_matrix = torch.from_numpy(np.array(mask_idx))
        all = self.layer(Variable(aim_matrix))  # bs*num_labels（最多混合人个数）×Embedding的大小
        out = all
        return out


class ADDJUST(nn.Module):
    # 这个模块是负责处理目标人的对应扰动的，进行一些偏移的调整
    def __init__(self, config, hidden_units, embedding_size):
        super(ADDJUST, self).__init__()
        self.config = config
        self.hidden_units = hidden_units
        self.emb_size = embedding_size
        self.layer = nn.Linear(hidden_units + embedding_size, embedding_size, bias=False)

    def forward(self, input_hidden, prob_emb):
        top_k_num = prob_emb.size()[1]
        x = torch.mean(input_hidden, 1).view(self.config.batch_size, 1, self.hidden_units).expand(
            self.config.batch_size, top_k_num, self.hidden_units)
        can = torch.cat([x, prob_emb], dim=2)
        all = self.layer(can)  # bs*num_labels（最多混合人个数）×Embedding的大小
        out = all
        return out


class SS(nn.Module):
    def __init__(self, config, speech_fre, mix_speech_len, num_labels):
        super(SS, self).__init__()
        self.config = config
        self.speech_fre = speech_fre
        self.mix_speech_len = mix_speech_len
        self.num_labels = num_labels
        print
        'Begin to build the maim model for speech speration part.'
        if config.speech_cnn_net:
            self.mix_hidden_layer_3d = MIX_SPEECH_CNN(config, speech_fre, mix_speech_len)
        else:
            self.mix_hidden_layer_3d = MIX_SPEECH(config, speech_fre, mix_speech_len)
        # att_layer=ATTENTION(config.EMBEDDING_SIZE,'dot')
        self.att_speech_layer = ATTENTION(config.EMBEDDING_SIZE, config.SPK_EMB_SIZE, config.ATT_SIZE, 'align')
        if self.config.is_SelfTune:
            self.adjust_layer = ADDJUST(config, 2 * config.HIDDEN_UNITS, config.SPK_EMB_SIZE)
            print
            'Adopt adjust layer.'

    def forward(self, mix_feas, hidden_outputs, targets, dict_spk2idx=None):
        '''
        :param targets:这个targets的大小是：topk,bs 注意后面要transpose
        123  324  345
        323  E     E
        这种样子的，所以要去找aim_list 应该找到的结果是,先transpose之后，然后flatten，然后取不是E的：[0 1 2 4 ]

        '''

        config = self.config
        top_k_max, batch_size = targets.size()  # 这个top_k_max其实就是最多有几个说话人，应该是跟Max_MIX是保持一样的
        # assert top_k_max==config.MAX_MIX
        aim_list = (targets.transpose(0, 1).contiguous().view(-1) != dict_spk2idx['<EOS>']).nonzero().squeeze()
        aim_list = aim_list.data.cpu().numpy()

        mix_speech_hidden, mix_tmp_hidden = self.mix_hidden_layer_3d(mix_feas)
        mix_speech_multiEmbs = torch.transpose(hidden_outputs, 0, 1).contiguous()  # bs*num_labels（最多混合人个数）×Embedding的大小
        mix_speech_multiEmbs = mix_speech_multiEmbs.view(-1, config.SPK_EMB_SIZE)  # bs*num_labels（最多混合人个数）×Embedding的大小
        # assert mix_speech_multiEmbs.size()[0]==targets.shape
        mix_speech_multiEmbs = mix_speech_multiEmbs[aim_list]  # aim_num,embs
        # mix_speech_multiEmbs=mix_speech_multiEmbs[0] # aim_num,embs
        # print mix_speech_multiEmbs.shape
        if self.config.is_SelfTune:
            # TODO: 这里应该也是有问题的，暂时不用selfTune
            mix_adjust = self.adjust_layer(mix_tmp_hidden, mix_speech_multiEmbs)
            mix_speech_multiEmbs = mix_adjust + mix_speech_multiEmbs
        mix_speech_hidden_5d = mix_speech_hidden.view(batch_size, 1, self.mix_speech_len, self.speech_fre,
                                                      config.EMBEDDING_SIZE)
        mix_speech_hidden_5d = mix_speech_hidden_5d.expand(batch_size, top_k_max, self.mix_speech_len, self.speech_fre,
                                                           config.EMBEDDING_SIZE).contiguous()
        mix_speech_hidden_5d_last = mix_speech_hidden_5d.view(-1, self.mix_speech_len, self.speech_fre,
                                                              config.EMBEDDING_SIZE)
        mix_speech_hidden_5d_last = mix_speech_hidden_5d_last[aim_list]
        att_multi_speech = self.att_speech_layer(mix_speech_hidden_5d_last,
                                                 mix_speech_multiEmbs.view(-1, config.SPK_EMB_SIZE))
        att_multi_speech = att_multi_speech.view(-1, self.mix_speech_len, self.speech_fre)  # bs,num_labels,len,fre这个东西
        multi_mask = att_multi_speech
        assert multi_mask.shape[0] == len(aim_list)
        return multi_mask


def top_k_mask(batch_pro, alpha, top_k):
    'batch_pro是 bs*n的概率分布，例如2×3的，每一行是一个概率分布\
    alpha是阈值，大于它的才可以取，可以跟Multi-label语音分离的ACC的alpha对应;\
    top_k是最多输出几个候选目标\
    输出是与bs*n的一个mask，float型的'
    size = batch_pro.size()
    final = torch.zeros(size)
    sort_result, sort_index = torch.sort(batch_pro, 1, True)  # 先排个序
    sort_index = sort_index[:, :top_k]  # 选出每行的top_k的id
    sort_result = torch.sum(sort_result > alpha, 1)
    for line_idx in range(size[0]):
        line_top_k = sort_index[line_idx][:int(sort_result[line_idx].data.cpu().numpy())]
        line_top_k = line_top_k.data.cpu().numpy()
        for i in line_top_k:
            final[line_idx, i] = 1
    return final


