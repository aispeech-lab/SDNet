# coding=utf8
import os
import csv
import codecs
import yaml
import time
import numpy as np
import shutil
import soundfile as sf
import librosa

from sklearn import metrics


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path, 'r')))


def read_datas(filename, trans_to_num=False):
    lines = open(filename, 'r').readlines()
    lines = list(map(lambda x: x.split(), lines))
    if trans_to_num:
        lines = [list(map(int, line)) for line in lines]
    return lines


def save_datas(data, filename, trans_to_str=False):
    if trans_to_str:
        data = [list(map(str, line)) for line in data]
    lines = list(map(lambda x: " ".join(x), data))
    with open(filename, 'w') as f:
        f.write("\n".join(lines))


def logging(file):
    def write_log(s):
        print(s, '')
        with open(file, 'a') as f:
            f.write(s)

    return write_log


def logging_csv(file):
    def write_csv(s):
        # with open(file, 'a', newline='') as f:
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(s)

    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def eval_metrics(reference, candidate, label_dict, log_path):
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)
    ref_file = ref_dir + 'reference'
    cand_file = cand_dir + 'candidate'

    for i in range(len(reference)):
        with codecs.open(ref_file + str(i), 'w', 'utf-8') as f:
            f.write("".join(reference[i]) + '\n')
        with codecs.open(cand_file + str(i), 'w', 'utf-8') as f:
            f.write("".join(candidate[i]) + '\n')

    def make_label(l, label_dict):
        length = len(label_dict)
        result = np.zeros(length)
        indices = [label_dict.get(label.strip().lower(), 0) for label in l]
        result[indices] = 1
        return result

    def prepare_label(y_list, y_pre_list, label_dict):
        reference = np.array([make_label(y, label_dict) for y in y_list])
        candidate = np.array([make_label(y_pre, label_dict) for y_pre in y_pre_list])
        return reference, candidate

    def get_metrics(y, y_pre):
        hamming_loss = metrics.hamming_loss(y, y_pre)
        macro_f1 = metrics.f1_score(y, y_pre, average='macro')
        macro_precision = metrics.precision_score(y, y_pre, average='macro')
        macro_recall = metrics.recall_score(y, y_pre, average='macro')
        micro_f1 = metrics.f1_score(y, y_pre, average='micro')
        micro_precision = metrics.precision_score(y, y_pre, average='micro')
        micro_recall = metrics.recall_score(y, y_pre, average='micro')
        return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall

    y, y_pre = prepare_label(reference, candidate, label_dict)
    hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall = get_metrics(y,
                                                                                                                 y_pre)
    return {'hamming_loss': hamming_loss,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall}


def bss_eval(config, predict_multi_map, y_multi_map, y_map_gtruth, train_data, dst='batch_output'):
    # dst='batch_output'
    if os.path.exists(dst):
        print(" \ncleanup: " + dst + "/")
        shutil.rmtree(dst)
    os.makedirs(dst)

    for sample_idx, each_sample in enumerate(train_data['multi_spk_wav_list']):
        for each_spk in each_sample.keys():
            this_spk = each_spk
            wav_genTrue = each_sample[this_spk]
            # min_len = 39936
            min_len = len(wav_genTrue)
            if config.FRAME_SHIFT == 64:
                min_len = len(wav_genTrue)
            sf.write(dst + '/{}_{}_realTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
                     config.FRAME_RATE, )

    predict_multi_map_list = []
    pointer = 0
    for each_line in y_map_gtruth:
        predict_multi_map_list.append(predict_multi_map[pointer:(pointer + len(each_line))])
        pointer += len(each_line)
    assert len(predict_multi_map_list) == len(y_map_gtruth)

    # 对于每个sample
    sample_idx = 0  # 代表一个batch里的依次第几个
    for each_y, each_pre, each_trueVector, spk_name in zip(y_multi_map, predict_multi_map_list, y_map_gtruth,
                                                           train_data['aim_spkname']):
        _mix_spec = train_data['mix_phase'][sample_idx]
        feas_tgt = train_data['multi_spk_fea_list'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for idx, one_cha in enumerate(each_trueVector):
            this_spk = one_cha
            y_pre_map = each_pre[idx].data.cpu().numpy()
            _pred_spec = y_pre_map * np.exp(1j * phase_mix)
            wav_pre = librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
            min_len = len(wav_pre)
            sf.write(dst + '/{}_{}_pre.wav'.format(sample_idx, this_spk), wav_pre[:min_len], config.FRAME_RATE, )

            gen_true_spec = feas_tgt[this_spk] * np.exp(1j * phase_mix)
            wav_gen_True = librosa.core.spectrum.istft(np.transpose(gen_true_spec), config.FRAME_SHIFT)
            sf.write(dst + '/{}_{}_genTrue.wav'.format(sample_idx, this_spk), wav_gen_True[:min_len],
                     config.FRAME_RATE, )
        sf.write(dst + '/{}_True_mix.wav'.format(sample_idx), train_data['mix_wav'][sample_idx][:min_len],
                 config.FRAME_RATE, )
        sample_idx += 1


def bss_eval2(config, predict_multi_map, y_multi_map, y_map_gtruth, train_data, dst='batch_output'):
    # dst='batch_output'
    if os.path.exists(dst):
        print(" \ncleanup: " + dst + "/")
        shutil.rmtree(dst)
    os.makedirs(dst)

    for sample_idx, each_sample in enumerate(train_data['multi_spk_wav_list']):
        for each_spk in each_sample.keys():
            this_spk = each_spk
            wav_genTrue = each_sample[this_spk]
            # min_len = 39936
            min_len = len(wav_genTrue)
            if config.FRAME_SHIFT == 64:
                min_len = len(wav_genTrue)
            sf.write(dst + '/{}_{}_realTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
                     config.FRAME_RATE, )

    predict_multi_map_list = []
    pointer = 0
    for each_line in y_map_gtruth:
        predict_multi_map_list.append(predict_multi_map[pointer:(pointer + len(each_line))])
        pointer += len(each_line)
    assert len(predict_multi_map_list) == len(y_map_gtruth)

    # 对于每个sample
    sample_idx = 0  # 代表一个batch里的依次第几个
    for each_y, each_pre, each_trueVector, spk_name in zip(y_multi_map, predict_multi_map_list, y_map_gtruth,
                                                           train_data['aim_spkname']):
        _mix_spec = train_data['mix_phase'][sample_idx]
        feas_tgt = train_data['multi_spk_fea_list'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        each_pre = each_pre[0]
        for idx, one_cha in enumerate(each_trueVector):
            this_spk = one_cha
            y_pre_map = each_pre[idx].data.cpu().numpy()
            _pred_spec = y_pre_map * np.exp(1j * phase_mix)
            wav_pre = librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
            min_len = len(wav_pre)
            sf.write(dst + '/{}_{}_pre.wav'.format(sample_idx, this_spk), wav_pre[:min_len], config.FRAME_RATE, )

            gen_true_spec = feas_tgt[this_spk] * np.exp(1j * phase_mix)
            wav_gen_True = librosa.core.spectrum.istft(np.transpose(gen_true_spec), config.FRAME_SHIFT)
            sf.write(dst + '/{}_{}_genTrue.wav'.format(sample_idx, this_spk), wav_gen_True[:min_len],
                     config.FRAME_RATE, )
        sf.write(dst + '/{}_True_mix.wav'.format(sample_idx), train_data['mix_wav'][sample_idx][:min_len],
                 config.FRAME_RATE, )
        sample_idx += 1

def bss_eval_tas(config, predict_wav, y_multi_map, y_map_gtruth, train_data, dst='batch_output'):
    # dst='batch_output'
    if os.path.exists(dst):
        print(" \ncleanup: " + dst + "/")
        shutil.rmtree(dst)
    os.makedirs(dst)

    for sample_idx, each_sample in enumerate(train_data['multi_spk_wav_list']):
        for each_spk in each_sample.keys():
            this_spk = each_spk
            wav_genTrue = each_sample[this_spk]
            # min_len = 39936
            min_len = len(wav_genTrue)
            if config.FRAME_SHIFT == 64:
                min_len = len(wav_genTrue)
            sf.write(dst + '/{}_{}_realTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
                     config.FRAME_RATE, )

    predict_multi_map_list = []
    pointer = 0
    # if len(predict_wav.shape)==3:
    #     predict_wav=predict_wav.view(-1,predict_wav.shape[-1])
    for each_line in y_map_gtruth:
        predict_multi_map_list.append(predict_wav[pointer:(pointer + len(each_line))])
        pointer += len(each_line)
    assert len(predict_multi_map_list) == len(y_map_gtruth)
    predict_multi_map_list=[i for i in predict_wav.unsqueeze(1)]

    # 对于每个sample
    sample_idx = 0  # 代表一个batch里的依次第几个
    for each_y, each_pre, each_trueVector, spk_name in zip(y_multi_map, predict_multi_map_list, y_map_gtruth,
                                                           train_data['aim_spkname']):
        _mix_spec = train_data['mix_phase'][sample_idx]
        feas_tgt = train_data['multi_spk_fea_list'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        each_pre = each_pre[0]
        for idx, one_cha in enumerate(each_trueVector):
            this_spk = one_cha
            y_pre_map = each_pre[idx].data.cpu().numpy()
            # _pred_spec = y_pre_map * np.exp(1j * phase_mix)
            # wav_pre = librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
            wav_pre = y_pre_map
            min_len = len(wav_pre)
            sf.write(dst + '/{}_{}_pre.wav'.format(sample_idx, this_spk), wav_pre[:min_len], config.FRAME_RATE, )

            gen_true_spec = feas_tgt[this_spk] * np.exp(1j * phase_mix)
            wav_gen_True = librosa.core.spectrum.istft(np.transpose(gen_true_spec), config.FRAME_SHIFT)
            sf.write(dst + '/{}_{}_genTrue.wav'.format(sample_idx, this_spk), wav_gen_True[:min_len],
                     config.FRAME_RATE, )
        sf.write(dst + '/{}_True_mix.wav'.format(sample_idx), train_data['mix_wav'][sample_idx][:min_len],
                 config.FRAME_RATE, )
        sample_idx += 1
