# coding=utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict


class global_attention(nn.Module):

    def __init__(self, hidden_size, activation=None):
        super(global_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size,bias=1)
        self.softmax = nn.Softmax()
        # self.batchnorm=nn.BatchNorm1d(hidden_size)
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, x, context):
        gamma_h = self.linear_in(x).unsqueeze(2)  # unsequeee这个函数相当于直接reshape多出来一维度，值得学习。   # batch * size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)  # batch * time
        weights = self.softmax(weights)  # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # batch * size
        output = self.linear_out(torch.cat([c_t, x], 1)) #添加额外的batchnorm
        # output = self.batchnorm(output)
        output = self.tanh(output)
        return output, weights
