#coding=utf8
import numpy as np
import os
import soundfile as sf
from separation import bss_eval_sources

path='batch_output/'
# path='/home/sw/Shin/Codes/DL4SS_Keras/TDAA_beta/batch_output2/'
def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    if src_ref.shape[0]==1:
        src_anchor=src_anchor[0]
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDR = ((sdr[0]) + (sdr[1])) / 2
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    return avg_SDR, avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """

    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi

def cal_SISNRi_PIT(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi) 2-mix
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """

    sisnr1_a = cal_SISNR(src_ref[0,:], src_est[0,:])
    sisnr2_a = cal_SISNR(src_ref[1,:], src_est[1,:])

    sisnr1_b = cal_SISNR(src_ref[0,:], src_est[1,:])
    sisnr2_b = cal_SISNR(src_ref[1,:], src_est[0,:])
    
    sisnr1_o = cal_SISNR(src_ref[0,:], mix)
    sisnr2_o = cal_SISNR(src_ref[1,:], mix)

    avg_SISNR = max((sisnr1_a+sisnr2_a)/2, (sisnr1_b+sisnr2_b)/2)
    avg_SISNRi = max( ((sisnr1_a - sisnr1_o) + (sisnr2_a - sisnr2_o)) / 2 ,((sisnr1_b - sisnr1_o) + (sisnr2_b - sisnr2_o)) / 2 )
    return avg_SISNR, avg_SISNRi

def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr

def cal(path,tmp=None):
    mix_number=len(set([l.split('_')[0] for l in os.listdir(path) if l[-3:]=='wav']))
    print('num of mixed :',mix_number)
    SDR_sum=np.array([])
    SDRi_sum=np.array([])
    for idx in range(mix_number):
        pre_speech_channel=[]
        aim_speech_channel=[]
        mix_speech=[]
        for l in sorted(os.listdir(path)):
            if l[-3:]!='wav':
                continue
            if l.split('_')[0]==str(idx):
                if 'True_mix' in l:
                    mix_speech.append(sf.read(path+l)[0])
                if 'real' in l and 'noise' not in l:
                    aim_speech_channel.append(sf.read(path+l)[0])
                if 'pre' in l:
                    pre_speech_channel.append(sf.read(path+l)[0])

        assert len(aim_speech_channel)==len(pre_speech_channel)
        aim_speech_channel=np.array(aim_speech_channel)
        pre_speech_channel=np.array(pre_speech_channel)
        mix_speech=np.array(mix_speech)
        assert mix_speech.shape[0]==1
        mix_speech=mix_speech[0]

        result=bss_eval_sources(aim_speech_channel,pre_speech_channel)
        SDR_sum=np.append(SDR_sum,result[0])

        # result=bss_eval_sources(aim_speech_channel,aim_speech_channel)
        # result_sdri=cal_SDRi(aim_speech_channel,pre_speech_channel,mix_speech)
        # print 'SDRi:',result_sdri
        result=cal_SISNRi(aim_speech_channel,pre_speech_channel,mix_speech)
        print('SI-SNR',result)
        # for ii in range(aim_speech_channel.shape[0]):
        #     result=cal_SISNRi(aim_speech_channel[ii],pre_speech_channel[ii],mix_speech[ii])
        #     print('SI-SNR',result)
        # SDRi_sum=np.append(SDRi_sum,result_sdri)

    print('SDR_Aver for this batch:',SDR_sum.mean())
    # print 'SDRi_Aver for this batch:',SDRi_sum.mean()
    return SDR_sum.mean(),SDRi_sum.mean()

# cal(path)

