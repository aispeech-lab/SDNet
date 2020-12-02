#coding=utf8
import torch
import torch.nn as nn
import soundfile as sf
import resampy
# from prepare_data_wsj2 import linearspectrogram
import librosa
import numpy as np
from models.istft_irfft import istft_irfft

# 参考https://github.com/jonlu0602/DeepDenoisingAutoencoder/blob/master/python/utils.py
def linearspectrogram(y, dBscale = 1, normalize=1):
    fft_size = 256
    hop_size = 128
    ref_db = 20
    max_db = 100
    D = librosa.core.spectrum.stft(y, fft_size, hop_size) # F, T
    F, T = D.shape
    S = np.abs(D)
    if dBscale:
        S = librosa.amplitude_to_db(S)
        if normalize:
            # normalization
            S = np.clip((S - ref_db + max_db) / max_db, 1e-8, 1)
    return S, np.angle(D)

def concatenateFeature(inputList, dim):
    out = inputList[0]
    for i in range(1, len(inputList)):
        out = torch.cat((out, inputList[i]), dim=dim)
    return out

class WaveLoss(nn.Module):
    def __init__(self, dBscale = 1, denormalize=1, max_db=100, ref_db=20, nfft=256, hop_size=128):
        super(WaveLoss, self).__init__()
        self.dBscale = dBscale
        self.denormalize = denormalize
        self.max_db = max_db
        self.ref_db = ref_db
        self.nfft = nfft
        self.hop_size = hop_size
        self.mse_loss = nn.MSELoss()

    def genWav(self, S, phase):
        '''
        :param S: (B, F-1, T) to be padded with 0 in this function
        :param phase: (B, F, T)
        :return: (B, num_samples)
        '''
        if self.dBscale:
            if self.denormalize:
                # denormalization
                S = S * self.max_db - self.max_db + self.ref_db
            # to amplitude
            # https://github.com/pytorch/pytorch/issues/12426
            # RuntimeError: the derivative for pow is not implemented
            # S = torch.pow(10, S * 0.05)
            S = 10 ** (S * 0.05)

        # pad with 0
        B, F, T = S.shape
        pad = torch.zeros(B, 1, T).to(S.device)
        # 注意tensor要同一类型
        Sfull = concatenateFeature([S, pad], dim=-2) # 由于前面预测少了一个维度的频率，所以这里补0

        # deal with the complex
        Sfull_ = Sfull.data.cpu().numpy()
        phase_ = phase.data.cpu().numpy()
        Sfull_spec = Sfull_ * np.exp(1.0j * phase_)
        S_sign = np.sign(np.real(Sfull_spec))
        S_sign = torch.from_numpy(S_sign).to(S.device)
        Sfull_spec_imag = np.imag(Sfull_spec)
        Sfull_spec_imag = torch.from_numpy(Sfull_spec_imag).unsqueeze(-1).to(S.device)
        Sfull = torch.mul(Sfull, S_sign).unsqueeze(-1)
        # print(Sfull.shape)
        # print(Sfull_spec_imag.shape)
        stft_matrix = concatenateFeature([Sfull, Sfull_spec_imag], dim=-1) # (B, F, T, 2)
        # print(stft_matrix.shape)

        wav = istft_irfft(stft_matrix, hop_length=self.hop_size, win_length=self.nfft)
        return wav

    def forward(self, target_mag, target_phase, pred_mag, pred_phase):
        '''
        :param target_mag: (B, F-1, T)
        :param target_phase: (B, F, T)
        :param pred_mag: (B, F-1, T)
        :param pred_phase: (B, F, T)
        :return:
        '''
        target_wav = self.genWav(target_mag, target_phase)
        pred_wav = self.genWav(pred_mag, pred_phase)

        # target_wav_arr = target_wav.squeeze(0).cpu().data.numpy()
        # pred_wav_arr = pred_wav.squeeze(0).cpu().data.numpy()
        # print('target wav arr', target_wav_arr.shape)
        # sf.write('target.wav', target_wav_arr, 8000)
        # sf.write('pred.wav', pred_wav_arr, 8000)

        loss = self.mse_loss(target_wav, pred_wav)
        return loss

if __name__ == '__main__':
    wav_f = 'test.wav'
    wav_fo = 'test_o.wav'
    def read_wav(f):
        wav, sr = sf.read(f)
        if len(wav.shape) > 1:
            wav = wav[:, 0]
        if sr != 8000:
            wav = resampy.resample(wav, sr, 8000)
        spec, phase = linearspectrogram(wav)
        return spec, phase, wav
    target_spec , target_phase, wav1 = read_wav(wav_f)
    pred_spec, pred_phase, wav2 = read_wav(wav_fo)
    # print('librosa,', pred_spec.shape)

    librosa_stft = librosa.stft(wav1, n_fft=256, hop_length=128, window='hann')
    # print('librosa stft', librosa_stft.shape)
    # print('librosa.stft', librosa_stft)
    _magnitude = np.abs(librosa_stft)
    # print('mag,', _magnitude.shape)
    wav_re_librosa = librosa.core.spectrum.istft(librosa_stft, hop_length=128)
    sf.write('wav_re_librosa.wav', wav_re_librosa, 8000)

    def clip_spec(spec, phase):
        spec_clip = spec[0:-1, :410] # (F-1, T)
        phase_clip = phase[:, :410]
        spec_tensor = torch.from_numpy(spec_clip)
        spec_tensor = spec_tensor.unsqueeze(0) # (B, T, F)
        # print(spec_tensor.shape)
        phase_tensor = torch.from_numpy(phase_clip).unsqueeze(0)
        # print(phase_tensor.shape)
        return spec_tensor, phase_tensor
    target_spec_tensor, target_phase_tensor = clip_spec(target_spec, target_phase)
    pred_spec_tensor, pred_phase_tensor = clip_spec(pred_spec, pred_phase)

    wav_loss = WaveLoss(dBscale=1, nfft=256, hop_size=128)
    loss = wav_loss(target_spec_tensor, target_phase_tensor, pred_spec_tensor, pred_phase_tensor)
    print('loss', loss.item())

    wav1 = torch.FloatTensor(wav1)
    torch_stft_matrix = torch.stft(wav1, n_fft=256, hop_length=128, window=torch.hann_window(256))
    torch_stft_matrix = torch_stft_matrix.unsqueeze(0)
    # print('torch stft', torch_stft_matrix.shape)
    # print(torch_stft_matrix[:,:,:,0])
    # print(torch_stft_matrix[:,:,:,1])
    wav_re = istft_irfft(torch_stft_matrix, hop_length=128, win_length=256)
    wav_re = wav_re.squeeze(0).cpu().data.numpy()
    # print('wav_re', wav_re.shape)
    sf.write('wav_re.wav', wav_re, 8000)