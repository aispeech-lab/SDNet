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
# config
parser = argparse.ArgumentParser(description='train_WSJ_tasnet.py')

parser.add_argument('-config', default='config_WSJ0_SDNet.yaml', type=str, help="config file")
parser.add_argument('-gpus', default=[0], nargs='+', type=int, help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str, help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234, help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str, help="Model selection")
parser.add_argument('-score', default='', type=str, help="score_fn")
parser.add_argument('-notrain', default=False, type=bool, help="train or not")
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

def train(epoch):
    global e, updates, total_loss, start_time, report_total,report_correct, total_loss_sgm, total_loss_ss,loss_last_epoch
    e = epoch
    model.train()
    SDR_SUM = np.array([])
    SDRi_SUM = np.array([])
    total_loss_final = 0

    if config.schedule and scheduler.get_lr()[0]>5e-5:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])

    if opt.model == 'gated':
        model.current_epoch = epoch

    train_data_gen = prepare_data('once', 'train')
    while True:
        train_data = next(train_data_gen)
        if train_data == False:
            print("ss loss (SISNR) in trainset:", total_loss_final)
            break 

        raw_tgt = train_data['batch_order']
        raw_tgt_dir = train_data['direction']

        padded_mixture, mixture_lengths, padded_source = train_data['tas_zip']
        padded_mixture = Variable(torch.from_numpy(padded_mixture).float())
        mixture_lengths = torch.from_numpy(mixture_lengths)
        padded_source = torch.from_numpy(padded_source).float()
        

        padded_mixture = padded_mixture.cuda().transpose(0,1)
        mixture_lengths = mixture_lengths.cuda()
        padded_source = padded_source.cuda()

        # 要保证底下这几个都是longTensor(长整数）
        tgt_max_len = config.MAX_MIX + 2  # with bos and eos.
        tgt = Variable(torch.from_numpy(np.array([[0] + [dict_spk2idx[spk] for spk in spks] + (tgt_max_len - len(spks) - 1) * [dict_spk2idx['<EOS>']] for
             spks in raw_tgt], dtype=np.int))).transpose(0, 1)  # 转换成数字，然后前后加开始和结束符号。
        tgt_dir = Variable(torch.from_numpy(np.array([[0] + [dict_dir2idx[int(dire)] for dire in directs] + (tgt_max_len - len(directs) - 1) * [dict_dir2idx['<EOS>']] for
             directs in raw_tgt_dir], dtype=np.int))).transpose(0, 1)  # 转换成数字，然后前后加开始和结束符号。
        src_len = Variable(torch.LongTensor(config.batch_size).zero_() + mix_speech_len).unsqueeze(0)
        tgt_len = Variable( torch.LongTensor([len(one_spk) for one_spk in train_data['multi_spk_fea_list']])).unsqueeze(0)
        if use_cuda:
            tgt = tgt.cuda()
            tgt_dir = tgt_dir.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
        
        model.zero_grad()

        # aim_list 就是找到有正经说话人的地方的标号
        aim_list = (tgt[1:-1].transpose(0, 1).contiguous().view(-1) != dict_spk2idx['<EOS>']).nonzero().squeeze()
        aim_list = aim_list.data.cpu().numpy()

        outputs, outputs_dir, targets, targets_dir, multi_mask = model(padded_mixture, tgt, tgt_dir)  # 这里的outputs就是hidden_outputs，还没有进行最后分类的隐层，可以直接用
        multi_mask = multi_mask.transpose(0,1)

        if 1 and len(opt.gpus) > 1:
            sgm_loss, num_total, num_correct, num_total_dir, num_correct_dir = model.module.compute_loss(outputs, outputs_dir, targets, targets_dir, opt.memory)
        else:
            sgm_loss, num_total, num_correct, num_total_dir, num_correct_dir = model.compute_loss(outputs, outputs_dir, targets, targets_dir, opt.memory)
        print('loss for SGM,this batch:', sgm_loss.item())
        print("num_total",num_total)
        print("num_correct",num_correct)

        if config.use_tas:
            if 1 and len(opt.gpus) > 1:
                ss_loss = model.module.separation_tas_loss(multi_mask, padded_source, mixture_lengths)
            else:
                ss_loss = model.separation_tas_loss(multi_mask, padded_source, mixture_lengths)

        print('loss for SS,this batch:', ss_loss.item())

        loss = 4*sgm_loss + ss_loss

        loss.backward()
        total_loss_sgm += sgm_loss.item()
        total_loss_ss += ss_loss.item()
        total_loss_final += ss_loss.item()
        
        report_correct += num_correct.item()
        report_total += num_total
        optim.step()
        updates += 1
        
        if updates % 30 == 0:
            logging(
                "time: %6.3f, epoch: %3d, updates: %8d, train loss this batch: %6.3f,sgm loss: %6.6f,ss loss: %6.6f,label acc: %6.6f\n"
                % (time.time() - start_time, epoch, updates, loss / num_total, total_loss_sgm / 30.0,
                   total_loss_ss / 30.0, report_correct/report_total))
            total_loss_sgm, total_loss_ss = 0, 0

    if total_loss_final < loss_last_epoch:
        loss_last_epoch = total_loss_final
        save_model(log_path + 'DSNet_{}_{}.pt'.format(epoch,total_loss_final)) 

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
    for i in range(1, config.epoch + 1):
            train(i)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
