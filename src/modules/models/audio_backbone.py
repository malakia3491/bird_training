import torch
import torch.nn as nn
import timm
import os

class AudioBackbone(nn.Module):
    def __init__(self, name, pretrained, in_chans, num_classes, global_pool, checkpoint_path=None):
        super().__init__()
        
        # Если есть локальный путь, отключаем авто-загрузку (pretrained=False)
        use_pretrained = pretrained and (checkpoint_path is None)
        
        self.model = timm.create_model(
            name,
            pretrained=use_pretrained,
            in_chans=in_chans,
            num_classes=0,       # Всегда 0, фичи only
            global_pool=global_pool
        )
        
        # Если передан путь к файлу - грузим вручную
        if checkpoint_path:
            self._load_custom_weights(checkpoint_path, in_chans)

        # Вычисляем размерность эмбеддинга
        with torch.no_grad():
            dummy = torch.randn(1, in_chans, 224, 224)
            out = self.model(dummy)
            self.embed_dim = out.shape[1]

    def _load_custom_weights(self, path, model_in_chans):
        print(f"Loading custom weights from: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # 1. Загрузка словаря весов
        if path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(path)
            except ImportError:
                raise ImportError("Please install safetensors: pip install safetensors")
        else:
            state_dict = torch.load(path, map_location="cpu")

        # 2. Адаптация первого слоя (3 канала -> 1 канал)
        # Модели ImageNet ждут 3 канала (RGB). У нас 1 (Спектрограмма).
        # Нам нужно найти первый слой в весах и усреднить/суммировать его веса.
        
        # Получаем текущее состояние модели
        model_dict = self.model.state_dict()
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                # Проверяем размерности
                model_shape = model_dict[k].shape
                checkpoint_shape = v.shape
                
                # Если размерности совпадают - просто копируем
                if model_shape == checkpoint_shape:
                    new_state_dict[k] = v
                
                # Если не совпадают во 2-м измерении (каналы): (N, 3, H, W) vs (N, 1, H, W)
                elif len(model_shape) == 4 and model_shape[1] == 1 and checkpoint_shape[1] == 3:
                    print(f"Adapting layer {k}: 3 channels -> 1 channel")
                    # Суммируем веса по оси каналов (стандартная практика timm)
                    new_state_dict[k] = v.sum(dim=1, keepdim=True)
                
                # Обработка трансформеров (Patch Embeddings)
                # ViT weights shape: (Nums, In_Chans * Patch*Patch) -> Flattened
                elif "patch_embed" in k and model_shape != checkpoint_shape:
                    # Это сложный случай для flatten весов, но timm обычно хранит их как Conv2d
                    # Если вдруг mismatch тут - просто пропускаем (инициализируем случайно)
                    print(f"Skipping layer {k} due to shape mismatch: {checkpoint_shape} vs {model_shape}")
                    continue
                else:
                    # Другие несовпадения (например голова классификации) пропускаем
                    pass
            else:
                pass # Лишние ключи (например head) игнорируем

        # 3. Загружаем веса (strict=False, т.к. мы могли выкинуть голову)
        self.model.load_state_dict(new_state_dict, strict=False)
        print("Custom weights loaded successfully.")

    def forward(self, x):
        return self.model(x)