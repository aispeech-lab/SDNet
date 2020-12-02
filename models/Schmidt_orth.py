# coding=utf8
import torch


def schmidt(this_vec, vectors):
    # this_vector是[bs,hidden_emb]
    # vectors是个列表，每一个里面应该是this_vector这么大的东西
    if len(vectors) == 0:
        return this_vec
    else:
        for vec in vectors:
            assert len(vec.size()) == len(this_vec.size()) == 2
            dot = torch.bmm(this_vec.unsqueeze(1), vec.unsqueeze(-1)).squeeze(-1)
            norm = torch.bmm(vec.unsqueeze(1), vec.unsqueeze(-1)).squeeze(-1)
            frac = dot / norm  # bs,1
            this_vec = this_vec - (frac * vec)
        # print 'final_vec:',this_vec
    return this_vec
