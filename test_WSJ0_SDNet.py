# coding=utf8
import os
import argparse
import time
import json
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np

import models
import data.utils as utils
from optims import Optim
import lr_scheduler as L
from predata_WSJ_lcx import prepare_data
import bss_test  
from models.loss import ss_tas_loss
from scipy.io import wavfile
# config
parser = argparse.ArgumentParser(description='train_WSJ_tasnet.py')

parser.add_argument('-config', default='config_WSJ0_SDNet.yaml', type=str, help="config file")
parser.add_argument('-gpus', default=[3], nargs='+', type=int, help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str, help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234, help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str, help="Model selection")
parser.add_argument('-score', default='', type=str, help="score_fn")
parser.add_argument('-notrain', default=True, type=bool, help="train or not")
parser.add_argument('-log', default='', type=str, help="log directory")
parser.add_argument('-memory', default=False, type=bool, help="memory efficiency")
parser.add_argument('-score_fc', default='', type=str, help="memory efficiency")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# checkpoint
if opt.restore:
    print('loading checkpoint...\n', opt.restore)
    checkpoints = torch.load(opt.restore,map_location={'cuda:2':'cuda:0'})

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
print(use_cuda)

# load the global statistic of the data
print('loading data...\n')
start_time = time.time()

spk_global_gen = prepare_data(mode='global', train_or_test='train')  # 数据中的一些统计参数的读取
global_para = next(spk_global_gen)
print(global_para)

spk_all_list = global_para['all_spk']  # 所有说话人的列表
dict_spk2idx = global_para['dict_spk_to_idx']
dict_idx2spk = global_para['dict_idx_to_spk']
direction_all_list = global_para['all_dir']
dict_dir2idx = global_para['dict_dir_to_idx']
dict_idx2dir = global_para['dict_idx_to_dir']
speech_fre = global_para['num_fre']  # 语音频率总数
total_frames = global_para['num_frames']  # 语音长度
spk_num_total = global_para['total_spk_num']  # 总计说话人数目
batch_total = global_para['total_batch_num']  # 一个epoch里多少个batch

print(dict_idx2spk)
print(dict_idx2dir)

config.speech_fre = speech_fre
mix_speech_len = total_frames
config.mix_speech_len = total_frames
num_labels = len(spk_all_list)
num_dir_labels = len(direction_all_list)

del spk_global_gen
print('loading the global setting cost: %.3f' % (time.time() - start_time))
print("num_dir_labels", num_dir_labels)
print("num_labels", num_labels)
# model
print('building model...\n')
model = getattr(models, opt.model)(config, 256, mix_speech_len, num_labels, num_dir_labels, use_cuda, None, opt.score_fc)

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if 0 and opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

optim.set_parameters(model.parameters())

if config.schedule:
    # scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    scheduler = L.StepLR(optim.optimizer, step_size=20, gamma=0.2)

# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

# logging modeule
if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
print('log_path:',log_path)

logging = utils.logging(log_path + 'log.txt')  # 单独写一个logging的函数，直接调用，既print，又记录到Log文件里。
logging_csv = utils.logging_csv(log_path + 'record.csv')
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model) + "\n\n")

logging('total number of parameters: %d\n\n' % param_count)
logging('score function is %s\n\n' % opt.score)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

total_loss, start_time = 0, time.time()
total_loss_sgm, total_loss_ss = 0, 0
report_total, report_correct = 0, 0
report_vocab, report_tot_vocab = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))
best_SDR = 0.0
e=0
loss_last_epoch = 1000000.0

def eval(epoch):
    # config.batch_size=1
    model.eval()
    # print '\n\n测试的时候请设置config里的batch_size为1！！！please set the batch_size as 1'
    reference, candidate, source, alignments = [], [], [], []
    e = epoch
    test_or_valid = 'test'
    #test_or_valid = 'valid'
    print('Test or valid:', test_or_valid)
    eval_data_gen = prepare_data('once', test_or_valid, config.MIN_MIX, config.MAX_MIX)
    SDR_SUM = np.array([])
    SDRi_SUM = np.array([])
    SISNR_SUM = np.array([])
    SISNRI_SUM = np.array([])
    SS_SUM = np.array([])
    batch_idx = 0
    global best_SDR, Var
    f = open('./results/spk2.txt', 'a')
    f_dir = open('./results/dir2.txt', 'a')
    f_bk = open('./results/spk_bk.txt', 'a')
    f_bk_dir = open('./results/dir_bk.txt', 'a')
    f_emb = open('./results/spk_emb.txt', 'a')
    f_emb_dir = open('./results/dir_emb.txt', 'a')
    f_hidden = open('./results/spk_hidden.txt', 'a')
    f_hidden_dir = open('./results/dir_hidden.txt', 'a')
    while True:
        print('-' * 30)
        eval_data =next(eval_data_gen)
        if eval_data == False:
            print('SDR_aver_eval_epoch:', SDR_SUM.mean())
            print('SDRi_aver_eval_epoch:', SDRi_SUM.mean())
            print('SISNR_aver_eval_epoch:', SISNR_SUM.mean())
            print('SISNRI_aver_eval_epoch:', SISNRI_SUM.mean())
            print('SS_aver_eval_epoch:', SS_SUM.mean())
            break  # 如果这个epoch的生成器没有数据了，直接进入下一个epoch

        raw_tgt= eval_data['batch_order']
        
        padded_mixture, mixture_lengths, padded_source = eval_data['tas_zip']
        padded_mixture=torch.from_numpy(padded_mixture).float()
        mixture_lengths=torch.from_numpy(mixture_lengths)
        padded_source=torch.from_numpy(padded_source).float()

        padded_mixture = padded_mixture.cuda().transpose(0,1)
        mixture_lengths = mixture_lengths.cuda()
        padded_source = padded_source.cuda()

        top_k = len(raw_tgt[0])
        tgt = Variable(torch.ones(top_k + 2, config.batch_size))
        src_len = Variable(torch.LongTensor(config.batch_size).zero_() + mix_speech_len).unsqueeze(0)
        tgt_len = Variable(torch.LongTensor([len(one_spk) for one_spk in eval_data['multi_spk_fea_list']])).unsqueeze(0)

        if use_cuda:
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()

        if 1 and len(opt.gpus) > 1:
            samples, samples_dir, alignment, hiddens, predicted_masks, output_list, output_dir_list, output_bk_list, output_dir_bk_list, hidden_list, hidden_dir_list, emb_list, emb_dir_list = model.module.beam_sample(padded_mixture, dict_spk2idx, dict_dir2idx, config.beam_size)
        else:
            samples, samples_dir, alignment, hiddens, predicted_masks, output_list, output_dir_list, output_bk_list, output_dir_bk_list, hidden_list, hidden_dir_list, emb_list, emb_dir_list = model.beam_sample(padded_mixture, dict_spk2idx,dict_dir2idx, config.beam_size)

        predicted_masks = predicted_masks.transpose(0,1)
        predicted_masks = predicted_masks[:,0:top_k,:] 
        mixture = torch.chunk(padded_mixture, 2, dim=-1)
        padded_mixture_c0 = mixture[0].squeeze()

        padded_source1= padded_source.data.cpu()
        predicted_masks1 = predicted_masks.data.cpu()

        padded_source= padded_source.squeeze().data.cpu().numpy()
        padded_mixture = padded_mixture.squeeze().data.cpu().numpy()
        predicted_masks = predicted_masks.squeeze().data.cpu().numpy()
        padded_mixture_c0 = padded_mixture_c0.squeeze().data.cpu().numpy()
        mixture_lengths = mixture_lengths.cpu()

        predicted_masks = predicted_masks - np.mean(predicted_masks)
        predicted_masks /= np.max(np.abs(predicted_masks))

        # '''''
        if batch_idx <= (3000 / config.batch_size):  # only the former batches counts the SDR
            
            sisnr, sisnri = bss_test.cal_SISNRi_PIT(padded_source, predicted_masks,padded_mixture_c0)
            sdr, sdri = bss_test.cal_SDRi(padded_source,predicted_masks, padded_mixture_c0)
            loss = ss_tas_loss(config,predicted_masks1, padded_source1, mixture_lengths, True)
            loss = loss.numpy()
            try:
                #SDR_SUM,SDRi_SUM = np.append(SDR_SUM, bss_test.cal('batch_output1/'))
                SDR_SUM = np.append(SDR_SUM, sdr)
                SDRi_SUM = np.append(SDRi_SUM, sdri)

                SISNR_SUM = np.append(SISNR_SUM, sisnr)
                SISNRI_SUM = np.append(SISNRI_SUM, sisnri)
                SS_SUM = np.append(SS_SUM, loss)
            except:# AssertionError,wrong_info:
                print('Errors in calculating the SDR',wrong_info)
            print('SDR_aver_now:', SDR_SUM.mean())
            print('SDRi_aver_now:', SDRi_SUM.mean())
            print('SISNR_aver_now:', SISNR_SUM.mean())
            print('SISNRI_aver_now:', SISNRI_SUM.mean())
            print('SS_aver_now:', SS_SUM.mean())

        elif batch_idx == (3000 / config.batch_size) + 1 and SDR_SUM.mean() > best_SDR:  # only record the best SDR once.
            print('Best SDR from {}---->{}'.format(best_SDR, SDR_SUM.mean()))
            best_SDR = SDR_SUM.mean()

        # '''
        candidate += [convertToLabels(dict_idx2spk, s, dict_spk2idx['<EOS>']) for s in samples]
        # source += raw_src
        reference += raw_tgt
        print('samples:', samples)
        print('can:{}, \nref:{}'.format(candidate[-1 * config.batch_size:], reference[-1 * config.batch_size:]))
        alignments += [align for align in alignment]
        batch_idx += 1
    f.close()
    f_dir.close()
    score = {}
    result = utils.eval_metrics(reference, candidate, dict_spk2idx, log_path)
    logging_csv([e, updates, result['hamming_loss'], \
                 result['micro_f1'], result['micro_precision'], result['micro_recall']])
    print('hamming_loss: %.8f | micro_f1: %.4f'
          % (result['hamming_loss'], result['micro_f1']))
    score['hamming_loss'] = result['hamming_loss']
    score['micro_f1'] = result['micro_f1']
    return score


# Convert `idx` to labels. If index `stop` is reached, convert it and return.
def convertToLabels(dict, idx, stop):
    labels = []

    for i in idx:
        i = int(i)
        if i == stop:
            break
        labels += [dict[i]]

    return labels


def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}

    torch.save(checkpoints, path)


def main():

    eval(1)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
