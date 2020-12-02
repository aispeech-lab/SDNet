# coding=utf8
import torch
import torch.nn as nn
import numpy as np
import data.dict as dict
from torch.autograd import Variable
import torch.nn.functional as F
import models.focal_loss as focal_loss
from itertools import permutations


EPS = 1e-8
def rank_feas(raw_tgt, feas_list, out_type='torch'):
    final_num = []
    for each_feas, each_line in zip(feas_list, raw_tgt):
        for spk in each_line:
            final_num.append(each_feas[spk])
    # 目标就是这个batch里一共有多少条比如 1spk 3spk 2spk,最后就是6个spk的特征
    if out_type=='numpy':
        return np.array(final_num)
    else:
        return torch.from_numpy(np.array(final_num))


def criterion(tgt_vocab_size, use_cuda, loss):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    if loss=='focal_loss':
        crit = focal_loss.FocalLoss(gamma=2)
    else:
        crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit

def criterion_dir(tgt_vocab_size, use_cuda, loss):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    if loss=='focal_loss':
        crit = focal_loss.FocalLoss(gamma=2)
    else:
        crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit


def memory_efficiency_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config):
    outputs = Variable(hidden_outputs.data, requires_grad=True, volatile=False)
    num_total, num_correct, loss = 0, 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = decoder.compute_score(out_t)
        loss_t = criterion(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(dict.PAD).data).sum()
        num_total_t = targ_t.ne(dict.PAD).data.sum()
        num_correct += num_correct_t
        num_total += num_total_t
        loss += loss_t.data[0]
        loss_t.div(num_total_t).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)

    return loss, num_total, num_correct, config.tgt_vocab, config.tgt_vocab


def cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, sim_score=0):
    # hidden_outputs:[max_len,bs,512]
    batch_size= targets.size()[1]
    targets=targets.view(-1)
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    scores = decoder.compute_score(outputs)
    loss = criterion(scores, targets.view(-1)) + sim_score
    pred = scores.max(1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    # num_correct = pred.data.eq(targets.data).masked_select(targets.ne(targets[-1]).data).sum()
    num_total = float(targets.ne(dict.PAD).data.sum())
    loss *= batch_size
    loss = loss.div(num_total)
    # loss = loss.data[0]

    return loss, num_total, num_correct

def cross_entropy_loss_dir(hidden_outputs, decoder, targets, criterion, config, sim_score=0):
    batch_size= targets.size()[1]
    targets=targets.view(-1)
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    scores = decoder.compute_score_dir(outputs)
    #print("scores:",scores.size())
    #print("targets:",targets.view(-1))
    loss = criterion(scores, targets.view(-1)) + sim_score
    pred = scores.max(1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = float(targets.ne(dict.PAD).data.sum())
    loss *= batch_size
    loss = loss.div(num_total)
    # loss = loss.data[0]

    return loss, num_total, num_correct

def mmse_loss(hidden_outputs, decoder, targets, mse_loss, softmax):
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    scores = softmax(decoder.compute_score_dir(outputs))
    targets = targets.view(-1)
    target_one_hot = torch.zeros(scores.size(0), dir_vocab_size).scatter_(1, targets, 1)
    #scores = linear(scores)
    print("score.size",scores)
    print("targets.size",targets)
    loss = mse_loss(scores.float(), targets.float())

    return loss

def mmse_loss2(hidden_outputs, decoder, targets, mse_loss):
    print("hidden_outputs", hidden_outputs.size())
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    scores_1, scores = decoder.compute_score_dir(outputs)
    scores = F.sigmoid(scores)
    print("score.size",scores)
    print("targets.size",targets)
    loss = mse_loss(scores.view(-1).float(), targets.view(-1).float()/20)

    return loss

def ss_loss(config, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func,wav_loss):
    predict_multi_map = multi_mask * x_input_map_multi
    # predict_multi_map=Variable(y_multi_map)
    y_multi_map = Variable(y_multi_map)

    loss_multi_speech = loss_multi_func(predict_multi_map, y_multi_map)

    # 各通道和为１的loss部分,应该可以更多的带来差异
    # y_sum_map=Variable(torch.ones(config.batch_size,config.mix_speech_len,config.speech_fre)).cuda()
    # predict_sum_map=torch.sum(multi_mask,1)
    # loss_multi_sum_speech=loss_multi_func(predict_sum_map,y_sum_map)
    print('loss 1 eval:  ', loss_multi_speech.data.cpu().numpy())
    # print('losssum eval :',loss_multi_sum_speech.data.cpu().numpy()
    # loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
    print('evaling multi-abs norm this eval batch:', torch.abs(y_multi_map - predict_multi_map).norm().data.cpu().numpy())
    # loss_multi_speech=loss_multi_speech+3*loss_multi_sum_speech
    print('loss for whole separation part:', loss_multi_speech.data.cpu().numpy())
    return loss_multi_speech

def ss_tas_loss(config,predict_wav, y_multi_wav, mix_length,loss_multi_func):
    loss = cal_loss_with_order(y_multi_wav, predict_wav, mix_length)[0]
    #loss_mse = loss_multi_func(predict_wav, y_multi_wav)
    return loss

def cal_loss_with_order(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    # print('real Tas SNI:',source[:,:,16000:16005])
    # print('pre Tas SNI:',estimate_source[:,:,16000:16005])
    max_snr = cal_si_snr_with_order(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(max_snr)
    return loss,

def cal_loss_with_PIT(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
                                                      estimate_source,
                                                      source_lengths)
    loss = 0 - torch.mean(max_snr)
    reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    return loss, max_snr, estimate_source, reorder_estimate_source

def cal_si_snr_with_order(source, estimate_source, source_lengths):
    """Calculate SI-SNR with given order.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    print("source.size()",source.size())
    print("estimate_source.size()",estimate_source.size())
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with order
    # reshape to use broadcast
    s_target = zero_mean_target  # [B, C, T]
    s_estimate = zero_mean_estimate  # [B, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]
    print(pair_wise_si_snr)

    return torch.sum(pair_wise_si_snr,dim=1)/C

def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    # perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    perms_one_hot = source.new_zeros((perms.size()[0],perms.size()[1], C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx

def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    # B, C, *_ = source.size()
    B, C, __ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    #mask = Variable(torch.ones((B, 1, T))).cuda()
    for i in range(B):
        #print("source_lengths[i]",source_lengths[i])
        mask[i, :, source_lengths[i]:] = 0
    return mask
def ss_loss_MLMSE(config, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func, Var):
    try:
        if Var == None:
            Var = Variable(torch.eye(config.speech_fre, config.speech_fre).cuda(), requires_grad=0)  # 初始化的是单位矩阵
            print('Set Var to:', Var)
    except:
        pass
    assert Var.size() == (config.speech_fre, config.speech_fre)

    predict_multi_map = torch.mean(multi_mask * x_input_map_multi, -2)  # 在时间维度上平均
    # predict_multi_map=Variable(y_multi_map)
    y_multi_map = torch.mean(Variable(y_multi_map), -2)  # 在时间维度上平均

    loss_vector = (y_multi_map - predict_multi_map).view(-1, config.speech_fre).unsqueeze(1)  # 应该是bs*1*fre

    Var_inverse = torch.inverse(Var)
    Var_inverse = Var_inverse.unsqueeze(0).expand(loss_vector.size()[0], config.speech_fre,
                                                  config.speech_fre)  # 扩展成batch的形式
    loss_multi_speech = torch.bmm(torch.bmm(loss_vector, Var_inverse), loss_vector.transpose(1, 2))
    loss_multi_speech = torch.mean(loss_multi_speech, 0)

    # 各通道和为１的loss部分,应该可以更多的带来差异
    y_sum_map = Variable(torch.ones(config.batch_size, config.mix_speech_len, config.speech_fre)).cuda()
    predict_sum_map = torch.sum(multi_mask, 1)
    loss_multi_sum_speech = loss_multi_func(predict_sum_map, y_sum_map)
    print('loss 1 eval, losssum eval : ', loss_multi_speech.data.cpu().numpy(), loss_multi_sum_speech.data.cpu().numpy())
    # loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
    print('evaling multi-abs norm this eval batch:', torch.abs(y_multi_map - predict_multi_map).norm().data.cpu().numpy())
    # loss_multi_speech=loss_multi_speech+3*loss_multi_sum_speech
    print('loss for whole separation part:', loss_multi_speech.data.cpu().numpy())
    # return F.relu(loss_multi_speech)
    return loss_multi_speech


def dis_loss(config, top_k_num, dis_model, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func):
    predict_multi_map = multi_mask * x_input_map_multi
    y_multi_map = Variable(y_multi_map).cuda()
    score_true = dis_model(y_multi_map)
    score_false = dis_model(predict_multi_map)
    acc_true = torch.sum(score_true > 0.5).data.cpu().numpy() / float(score_true.size()[0])
    acc_false = torch.sum(score_false < 0.5).data.cpu().numpy() / float(score_true.size()[0])
    acc_dis = (acc_false + acc_true) / 2
    print('acc for dis:(ture,false,aver)', acc_true, acc_false, acc_dis)

    loss_dis_true = loss_multi_func(score_true, Variable(torch.ones(config.batch_size * top_k_num, 1)).cuda())
    loss_dis_false = loss_multi_func(score_false, Variable(torch.zeros(config.batch_size * top_k_num, 1)).cuda())
    loss_dis = loss_dis_true + loss_dis_false
    print('loss for dis:(ture,false)', loss_dis_true.data.cpu().numpy(), loss_dis_false.data.cpu().numpy())
    return loss_dis
