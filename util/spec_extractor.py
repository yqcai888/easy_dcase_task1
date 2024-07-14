import numpy as np
import torch
import torchaudio
from torch import nn
import torchaudio.compliance.kaldi as ta_kaldi
import dcase_util

class _SpecExtractor(nn.Module):
    """ Base Module for spectrogram extractors. """

class Cnn3Mel(_SpecExtractor):
    """ Mel extractor for previous CNN3 baseline system. """
    def __init__(self,
                 spectrogram_type="magnitude",
                 hop_length_seconds=0.02,
                 win_length_seconds=0.04,
                 window_type="hamming_asymmetric",
                 n_mels=40,
                 n_fft=2048,
                 fmin=0,
                 fmax=22050,
                 htk=False,
                 normalize_mel_bands=False,
                 **kwargs):
        super().__init__()
        self.extractor = dcase_util.features.MelExtractor(spectrogram_type=spectrogram_type,
                                                          hop_length_seconds=hop_length_seconds,
                                                          win_length_seconds=win_length_seconds,
                                                          window_type=window_type,
                                                          n_mels=n_mels,
                                                          n_fft=n_fft,
                                                          fmin=fmin,
                                                          fmax=fmax,
                                                          htk=htk,
                                                          normalize_mel_bands=normalize_mel_bands,
                                                          **kwargs)

    def forward(self, x):
        mel = []
        for wav in x:
            wav = wav.cpu().numpy()
            mel.append(self.extractor.extract(wav))
        mel = np.stack(mel)
        mel = torch.from_numpy(mel).to(x.device)
        return mel


class CpMel(_SpecExtractor):
    """
    Mel extractor for CP-JKU systems. Adapted from: https://github.com/fschmid56/cpjku_dcase23
    """
    def __init__(self, n_mels=256, sr=32000, win_length=3072, hop_size=500, n_fft=4096, fmin=0.0, fmax=None):
        super().__init__()
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax
        self.hop_size = hop_size
        self.register_buffer('window', torch.hann_window(win_length, periodic=False),
                             persistent=False)
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient.to(x.device)).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hop_size, win_length=self.win_length,
                       center=True, normalized=False, window=self.window.to(x.device), return_complex=True)
        x = torch.view_as_real(x)
        x = (x ** 2).sum(dim=-1)  # power mag
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr,
                                                                 self.fmin, self.fmax, vtln_low=100.0, vtln_high=-500.,
                                                                 vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)
        # Log mel spectrogram
        melspec = (melspec + 0.00001).log()
        # Fast normalization
        melspec = (melspec + 4.5) / 5.
        return melspec


class BEATsMel(_SpecExtractor):
    """ Mel extractor for BEATs model. """
    def __init__(self, dataset_mean: float = 15.41663, dataset_std: float = 6.55582):
        super(BEATsMel, self).__init__()
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

    def forward(self, x):
        fbanks = []
        for waveform in x:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.dataset_mean) / (2 * self.dataset_std)
        return fbank