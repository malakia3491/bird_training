import torch
import torch.nn as nn
import torchaudio

class PCENFrontend(nn.Module):
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
            power=1.0 # Для PCEN нужна амплитуда, не мощность
        )
        
        # Параметры PCEN (делаем их обучаемыми!)
        # s (smoothing) обычно константа, но остальное можно учить
        self.s = cfg.time_constant
        self.eps = cfg.eps
        
        # Обучаемые параметры на каждый мелл-канал
        self.alpha = nn.Parameter(torch.full((cfg.n_mels, 1), cfg.gain)) # gain in config
        self.delta = nn.Parameter(torch.full((cfg.n_mels, 1), cfg.bias)) # bias in config
        self.r = nn.Parameter(torch.full((cfg.n_mels, 1), cfg.power))    # power in config

    def pcen(self, E):
        # E shape: (Batch, Freq, Time)
        
        # 1. Temporal Integration (IIR Filter)
        # M[t] = (1-s)*M[t-1] + s*E[t]
        # В PyTorch нет быстрого IIR, используем приближение или простой цикл (он быстрый по времени)
        # Для батчей эффективнее использовать custom kernel, но для старта цикл сойдет.
        
        M = torch.zeros_like(E)
        m_prev = E[:, :, 0]
        M[:, :, 0] = m_prev
        
        # Разворачиваем цикл сглаживания (это узкое место, но на GPU работает сносно для 5 сек)
        # Можно заменить на torchaudio.functional.filtfilt если подготовить коэффициенты
        for t in range(1, E.shape[2]):
            m_prev = (1 - self.s) * m_prev + self.s * E[:, :, t]
            M[:, :, t] = m_prev

        # 2. AGC (Automatic Gain Control)
        # PCEN = (E / (eps + M)^alpha + delta)^r - delta^r
        
        # Бродкастинг параметров (Freq, 1) -> (Batch, Freq, Time)
        alpha = self.alpha.unsqueeze(0)
        delta = self.delta.unsqueeze(0)
        r = self.r.unsqueeze(0)
        
        smooth = (self.eps + M).pow(alpha)
        pcen = (E / smooth + delta).pow(r) - delta.pow(r)
        
        return pcen

    def forward(self, x):
        # 1. Mel (Energy)
        mel = self.mel_transform(x) 
        
        # 2. PCEN
        spec = self.pcen(mel)
        
        return spec.unsqueeze(1)