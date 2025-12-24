import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """
        features: [Batch_Size, Dim] (уже склеенные 2N, нормализованные)
        labels: [Batch_Size] (уже склеенные 2N)
        """
        device = features.device

        # Если features пришли как [N, 2, Dim] -> схлопываем в [2N, Dim]
        if len(features.shape) == 3:
            features = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Если features пришли как [2N, 1, Dim] -> схлопываем в [2N, Dim]
        if len(features.shape) == 3 and features.shape[1] == 1:
            features = features.squeeze(1)

        batch_size = features.shape[0]

        # --- DEBUG БЛОК ---
        if labels is not None:
            # Выпрямляем labels
            labels = labels.view(-1)
            
            # АВТО-ФИКС: Если меток в 2 раза меньше чем фичей (N vs 2N), дублируем метки
            if labels.shape[0] == batch_size // 2:
                labels = torch.cat([labels, labels], dim=0)
            
            # Если всё равно не совпадает - падаем с подробностями
            if labels.shape[0] != batch_size:
                raise ValueError(
                    f"CRITICAL SHAPE MISMATCH in SupConLoss:\n"
                    f"Features (Batch Size): {batch_size}\n"
                    f"Labels: {labels.shape[0]}\n"
                    f"Check your SupConSystem.training_step concatenation logic."
                )

        # 1. Матрица сходства (Cosine Similarity)
        # features уже должны быть L2-нормализованы снаружи!
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # Численная стабильность
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 2. Создание маски
        if labels is not None:
            # 1 если класс совпадает, 0 если нет
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # Fallback (SimCLR mode): только диагональ
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        # 3. Убираем диагональ (сам с собой)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 4. Лог-вероятности
        exp_logits = torch.exp(logits) * logits_mask
        # Сумма экспонент по строке (знаменатель)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # 5. Считаем лосс
        # Сумма лог-вероятностей для всех позитивов
        mean_log_prob_pos = (mask * log_prob).sum(1)
        
        # Делим на количество позитивов
        # (Если позитивов нет, ставим 1 чтобы не делить на 0, потом результат занулим)
        pos_per_sample = mask.sum(1)
        pos_per_sample = torch.where(pos_per_sample > 0, pos_per_sample, torch.ones_like(pos_per_sample))
        
        mean_log_prob_pos = mean_log_prob_pos / pos_per_sample

        # Учитываем только те примеры, где БЫЛИ позитивные пары
        has_positives = mask.sum(1) > 0
        
        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = - mean_log_prob_pos[has_positives].mean()
        
        return loss