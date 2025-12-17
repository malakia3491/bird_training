import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (для SimCLR).
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: (Batch, Dim) - уже нормализованные проекции
        batch_size = z1.shape[0]
        
        # Конкатенируем: (2N, Dim) -> [z1_1, z1_2... z2_1, z2_2...]
        out = torch.cat([z1, z2], dim=0)
        
        # Матрица схожести (Cosine Similarity)
        # sim[i, j] = z_i * z_j / temperature
        # Мы предполагаем, что z уже нормализованы (L2 norm)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        
        # Маска, чтобы убрать схожесть элемента с самим собой (диагональ)
        mask = ~torch.eye(2 * batch_size, device=sim.device).bool()
        
        # Для каждого элемента i:
        # Положительная пара (positive) - это z1[i] и z2[i]
        # Отрицательные пары (negatives) - все остальные
        
        # Создаем метки. Для z1[k] пара это z2[k] (который находится по индексу k + batch_size)
        # Для z2[k] пара это z1[k] (индекс k)
        
        # Но проще реализовать через CrossEntropy
        # Позитивные пары лежат на диагоналях смещенных матриц
        
        # Чслитель: exp(sim(z1, z2) / t)
        # Знаменатель: sum(exp(sim(z1, all) / t))
        
        # Реализация через CrossEntropy (стандарт SimCLR):
        # target для элемента i (из 2N) - это индекс его пары.
        
        # Это сложная матричная магия, вот проверенная реализация:
        labels = torch.arange(batch_size, device=z1.device, dtype=torch.long)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Матрица логитов (2N, 2N)
        # Убираем диагональ (самих себя) из расчетов лосса сложно в одной матрице.
        # Стандартная реализация SimCLR делает чуть иначе, но давай простую версию:
        
        # SimCLR logic:
        # Для i из первой половины (z1): правильный класс = i + batch_size
        # Для i из второй половины (z2): правильный класс = i - batch_size (или просто i % batch)
        
        logits = cov / self.temperature
        
        # Нам нужно занулить (сделать -inf) диагональ, чтобы softmax не выбрал сам себя
        mask_diag = torch.eye(2 * batch_size, device=z1.device).bool()
        logits.masked_fill_(mask_diag, -9e15)
        
        loss = F.cross_entropy(logits, labels)
        return loss