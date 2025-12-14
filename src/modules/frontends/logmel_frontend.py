import torch
import torch.nn as nn
import torchaudio

class LogMelFrontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=cfg.power,
            normalized=True
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def forward(self, x):
        # x shape: (Batch, Time)
        # 1. Mel Spectrogram
        spec = self.mel_transform(x)
        
        # 2. Log conversion (dB)
        spec = self.amplitude_to_db(spec)
        
        # 3. Add Channel dimension: (Batch, Freq, Time) -> (Batch, 1, Freq, Time)
        return spec.unsqueeze(1)