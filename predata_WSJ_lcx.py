# coding=utf8
import os
import numpy as np
import random
import re
import soundfile as sf
import resampy
import librosa
import argparse
import data.utils as utils
import models
from scipy.io import wavfile
import scipy.signal
# Add the config.
parser = argparse.ArgumentParser(description='predata scripts.')
parser.add_argument('-config', default='config_WSJ0_Tasnet.yaml', type=str, help="config file")
opt = parser.parse_args()
config = utils.read_config(opt.config)

channel_first = config.channel_first
np.random.seed(1) 
random.seed(1)

data_path = '/mnt/lustre/xushuang2/lcx/data/amcc-data/2channel'

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def get_energy_order(multi_spk_fea_list):
    order=[]
    for one_line in multi_spk_fea_list:
        dd=sorted(one_line.items(),key= lambda d:d[1].sum(),reverse=True)
        dd=[d[0] for d in dd]
        order.append(dd)
    return order

def get_spk_order(dir_tgt,raw_tgt):
    raw_tgt_dir=[]
    i=0
    for sample in raw_tgt:
        dd = [dir_tgt[i][spk] for spk in sample]
        raw_tgt_dir.append(dd)
        i=i+1
    return raw_tgt_dir


def _collate_fn(mix_data,source_data,raw_tgt=None):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    mixtures, sources = mix_data,source_data
    if raw_tgt is None: #如果没有给定顺序
        raw_tgt = [sorted(spk.keys()) for spk in source_data]
    # sources= models.rank_feas(raw_tgt, source_data,out_type='numpy') 
    sources=[]
    for each_feas, each_line in zip(source_data, raw_tgt):
        sources.append(np.stack([each_feas[spk] for spk in each_line]))
    sources=np.array(sources)
    mixtures=np.array(mixtures)
    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    # mixtures_pad = pad_list([mix.float() for mix in mixtures], pad_value)
    ilens = ilens
    # sources_pad = pad_list([torch.from_numpy(s).float() for s in sources], pad_value)
    # N x T x C -> N x C x T
    # sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    return mixtures, ilens, sources
    # return mixtures_pad, ilens, sources_pad

def prepare_data(mode, train_or_test, min=None, max=None):

    if min:
        config.MIN_MIX = min
    if max:
        config.MAX_MIX = max

    mix_speechs = [] 
    aim_fea = []  
    aim_spkid = [] 
    aim_spkname = []  
    query = []  
    multi_spk_fea_list = []
    multi_spk_wav_list = []
    direction = []

    if config.MODE == 1:
        if config.DATASET == 'WSJ0':
            spk_file_tr = open('/mnt/lustre/xushuang2/lcx/data/amcc-data/2channel/wav_spk.txt','r')
            all_spk_train = [i.replace("\n","") for i in spk_file_tr]
            all_spk_train = sorted(all_spk_train)
            print(all_spk_train) 

            spk_file_tt = open('/mnt/lustre/xushuang2/lcx/data/amcc-data/2channel/test/wav_spk.txt','r')
            all_spk_test = [i.replace("\n","") for i in spk_file_tt]
            all_spk_test = sorted(all_spk_test)
            print(all_spk_test)
            all_spk = all_spk_train + all_spk_test
            print(all_spk)

            all_dir = [i for i in range(1,20)]
            dicDirFile = open('/mnt/lustre/xushuang2/lcx/data/amcc-data/2channel/wav_dirLabel2.txt', 'r')#打开数据  
            dirDict = {} 
            while True:  
                line = dicDirFile.readline()  
                if line == '':  
                    break  
                index = line.find(' ')
                key = line[:index]
                #print(key)  
                value = line[index:]  
                dirDict[key] = value.replace("\n","").replace(" ","")
            dicDirFile.close()
            
            spk_samples_list = {}
            batch_idx = 0
            list_path = '/mnt/lustre/xushuang2/lcx/data/create-speaker-mixtures/'
            all_samples_list = {}
            sample_idx = {}
            number_samples = {}
            batch_mix = {}
            mix_number_list = range(config.MIN_MIX, config.MAX_MIX + 1)
            number_samples_all = 0
            for mix_k in mix_number_list:
                if train_or_test == 'train':
                    aim_list_path = list_path + 'mix_{}_spk_tr.txt'.format(mix_k)
                if train_or_test == 'valid':
                    aim_list_path = list_path + 'mix_{}_spk_cv.txt'.format(mix_k)
                if train_or_test == 'test':
                    aim_list_path = list_path + 'mix_{}_spk_tt.txt'.format(mix_k)
                    config.batch_size = 1

                all_samples_list[mix_k] = open(aim_list_path).readlines()  # [:31]
                number_samples[mix_k] = len(all_samples_list[mix_k])
                batch_mix[mix_k] = len(all_samples_list[mix_k]) / config.batch_size
                number_samples_all += len(all_samples_list[mix_k])

                sample_idx[mix_k] = 0

                if train_or_test == 'train' and config.SHUFFLE_BATCH:
                    random.shuffle(all_samples_list[mix_k])
                    print('shuffle success!', all_samples_list[mix_k][0])

            batch_total = number_samples_all / config.batch_size
            
            mix_k = random.sample(mix_number_list, 1)[0]
            # while True:
            for ___ in range(number_samples_all):
                if ___ == number_samples_all - 1:
                    print('ends here.___')
                    yield False
                mix_len = 0
                if sample_idx[mix_k] >= batch_mix[mix_k] * config.batch_size:
                    mix_number_list.remove(mix_k)
                    try:
                        mix_k = random.sample(mix_number_list, 1)[0]
                    except ValueError:
                        print('seems there gets all over.')
                        if len(mix_number_list) == 0:
                            print('all mix number is over~!')
                        yield False
                    
                    batch_idx = 0
                    mix_speechs = np.zeros((config.batch_size, config.MAX_LEN))
                    mix_feas = [] 
                    mix_phase = []
                    aim_fea = []  
                    aim_spkid = [] 
                    aim_spkname = []
                    query = []  
                    multi_spk_fea_list = []
                    multi_spk_order_list=[] 
                    multi_spk_wav_list = []
                    continue

                all_over = 1  
                for kkkkk in mix_number_list:
                    if not sample_idx[kkkkk] >= batch_mix[mix_k] * config.batch_size:
                        all_over = 0
                        break
                    if all_over:
                        print('all mix number is over~!')
                        yield False

                # mix_k=random.sample(mix_number_list,1)[0]
                if train_or_test == 'train':
                    aim_spk_k = random.sample(all_spk_train, mix_k)  
                elif train_or_test == 'test':
                    aim_spk_k = random.sample(all_spk_test, mix_k)  

                aim_spk_k = re.findall('/([0-9][0-9].)/', all_samples_list[mix_k][sample_idx[mix_k]])
                aim_spk_db_k = [float(dd) for dd in re.findall(' (.*?) ', all_samples_list[mix_k][sample_idx[mix_k]])]
                aim_spk_samplename_k = re.findall('/(.{8})\.wav ', all_samples_list[mix_k][sample_idx[mix_k]])
                assert len(aim_spk_k) == mix_k == len(aim_spk_db_k) == len(aim_spk_samplename_k)

                multi_fea_dict_this_sample = {}
                multi_wav_dict_this_sample = {}
                multi_name_list_this_sample = []
                multi_db_dict_this_sample = {}
                direction_sample = {}
                for k, spk in enumerate(aim_spk_k):
                    
                    sample_name = aim_spk_samplename_k[k]
                    if aim_spk_db_k[k] ==0:
                        aim_spk_db_k[k] = int(aim_spk_db_k[k])
                    if train_or_test != 'test':
                        spk_speech_path = data_path + '/' + 'train' + '/' + sample_name + '_' +str(aim_spk_db_k[k])+ '_simu_nore.wav'
                    else:
                        spk_speech_path = data_path + '/' + 'test' + '/' + sample_name + '_' +str(aim_spk_db_k[k])+ '_simu_nore.wav'

                    signal, rate = sf.read(spk_speech_path)

                    wav_name = sample_name+ '_' +str(aim_spk_db_k[k])+ '_simu_nore.wav'
                    direction_sample[spk] = dirDict[wav_name]
                    if rate != config.FRAME_RATE:
                        print("config.FRAME_RATE",config.FRAME_RATE)
                        signal = signal.transpose()
                        signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
                        signal = signal.transpose()
                    
                    if signal.shape[0] > config.MAX_LEN:  
                        signal = signal[:config.MAX_LEN,:]

                    if signal.shape[0] > mix_len:
                        mix_len = signal.shape[0]

                    signal -= np.mean(signal)  
                    signal /= np.max(np.abs(signal))  

                    if signal.shape[0] < config.MAX_LEN:
                        signal = np.r_[signal, np.zeros((config.MAX_LEN - signal.shape[0],signal.shape[1]))]

                    if k == 0:  
                        ratio = 10 ** (aim_spk_db_k[k] / 20.0)
                        signal = ratio * signal
                        aim_spkname.append(aim_spk_k[0])
                        aim_spk_speech = signal
                        aim_spkid.append(aim_spkname)
                        wav_mix = signal
                        signal_c0 = signal[:,0]
                        a,b,frq = scipy.signal.stft(signal_c0, fs=8000, nfft=config.FRAME_LENGTH, noverlap=config.FRAME_SHIFT)
                        aim_fea_clean = np.transpose(np.abs(frq))
                        aim_fea.append(aim_fea_clean)
                        multi_fea_dict_this_sample[spk] = aim_fea_clean
                        multi_wav_dict_this_sample[spk] = signal[:,0]

                    else:
                        ratio = 10 ** (aim_spk_db_k[k] / 20.0)
                        signal = ratio * signal
                        wav_mix = wav_mix + signal  
                        a,b,frq = scipy.signal.stft(signal[:,0], fs=8000, nfft=config.FRAME_LENGTH, noverlap=config.FRAME_SHIFT)
                        some_fea_clean = np.transpose(np.abs(frq))
                        multi_fea_dict_this_sample[spk] = some_fea_clean
                        multi_wav_dict_this_sample[spk] = signal[:,0]

                multi_spk_fea_list.append(multi_fea_dict_this_sample)  
                multi_spk_wav_list.append(multi_wav_dict_this_sample) 

                mix_speechs.append(wav_mix)
                direction.append(direction_sample)
                batch_idx += 1
            
                if batch_idx == config.batch_size:  
                    mix_k = random.sample(mix_number_list, 1)[0]
                    aim_fea = np.array(aim_fea)
                    query = np.array(query)
                    print('spk_list_from_this_gen:{}'.format(aim_spkname))
                    print('aim spk list:', [one.keys() for one in multi_spk_fea_list])
                    batch_ordre=get_energy_order(multi_spk_wav_list)
                    direction = get_spk_order(direction, batch_ordre)
                    if mode == 'global':
                        all_spk = sorted(all_spk)
                        all_spk = sorted(all_spk_train)
                        all_spk.insert(0, '<BOS>')  # 添加两个结构符号，来标识开始或结束。
                        all_spk.append('<EOS>')
                        all_dir = sorted(all_dir)
                        all_dir.insert(0, '<BOS>')
                        all_dir.append('<EOS>')
                        all_spk_test = sorted(all_spk_test)
                        dict_spk_to_idx = {spk: idx for idx, spk in enumerate(all_spk)}
                        dict_idx_to_spk = {idx: spk for idx, spk in enumerate(all_spk)}
                        dict_dir_to_idx = {dire: idx for idx, dire in enumerate(all_dir)}
                        dict_idx_to_dir = {idx: dire for idx, dire in enumerate(all_dir)}
                        yield {'all_spk': all_spk,
                               'dict_spk_to_idx': dict_spk_to_idx,
                               'dict_idx_to_spk': dict_idx_to_spk,
                               'all_dir': all_dir,
                               'dict_dir_to_idx': dict_dir_to_idx,
                               'dict_idx_to_dir': dict_idx_to_dir,
                               'num_fre': aim_fea.shape[2],  
                               'num_frames': aim_fea.shape[1],  
                               'total_spk_num': len(all_spk),
                               'total_batch_num': batch_total
                               }
                    elif mode == 'once':
                        yield {'mix_wav': mix_speechs,
                               'aim_fea': aim_fea,
                               'aim_spkname': aim_spkname,
                               'direction': direction,
                               'query': query,
                               'num_all_spk': len(all_spk),
                               'multi_spk_fea_list': multi_spk_fea_list,
                               'multi_spk_wav_list': multi_spk_wav_list,
                               'batch_order': batch_ordre,
                               'batch_total': batch_total,
                               'tas_zip': _collate_fn(mix_speechs,multi_spk_wav_list,batch_ordre)
                               }
                    elif mode == 'tasnet':
                        yield _collate_fn(mix_speechs,multi_spk_wav_list)

                    batch_idx = 0
                    mix_speechs = []
                    aim_fea = []  
                    aim_spkid = []  
                    aim_spkname = []
                    query = []  
                    multi_spk_fea_list = []
                    multi_spk_wav_list = []
                    direction = []
                sample_idx[mix_k] += 1

        else:
            raise ValueError('No such dataset:{} for Speech.'.format(config.DATASET))
        pass

    else:
        raise ValueError('No such Model:{}'.format(config.MODE))

if __name__ == '__main__':
    train_len=[]
    train_data_gen = prepare_data('once', 'train')
    while True:
        train_data_gen.next()
    pass
    print(np.array(train_len).mean())
