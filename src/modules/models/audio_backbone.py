import torch
import torch.nn as nn
import timm
import os

class AudioBackbone(nn.Module):
    def __init__(self, name, pretrained, in_chans, num_classes, global_pool, checkpoint_path=None, **kwargs):
        super().__init__()
        
        # Если передан путь, отключаем авто-загрузку ImageNet весов
        use_pretrained = pretrained and (checkpoint_path is None)
        
        self.model = timm.create_model(
            name,
            pretrained=use_pretrained,
            in_chans=in_chans,
            num_classes=0,       # Убираем голову
            global_pool=global_pool 
        )
        
        # Если есть кастомный чекпоинт (SSL/DAE)
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

        # Поддержка safetensors и pth
        if path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(path)
            except ImportError:
                raise ImportError("Please install safetensors: pip install safetensors")
        else:
            # Важно: weights_only=False для совместимости с Lightning чекпоинтами
            state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Если это Lightning чекпоинт, веса могут быть внутри "state_dict"
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model_dict = self.model.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # Очистка ключей (backbone.model. -> model. -> "")
            # Мы хотим, чтобы ключи совпадали с self.model (timm)
            
            # 1. Убираем префикс backbone, если есть
            clean_k = k.replace("backbone.", "")
            
            # 2. У timm ключи обычно сразу идут как "conv1...", но если мы сохраняли
            # через AudioBackbone, то они могут быть "model.conv1..."
            if clean_k.startswith("model."):
                clean_k = clean_k.replace("model.", "")
                
            if clean_k in model_dict:
                # Проверка размерностей (особенно для 1 канала)
                model_shape = model_dict[clean_k].shape
                checkpoint_shape = v.shape
                
                if model_shape == checkpoint_shape:
                    new_state_dict[clean_k] = v
                elif len(model_shape) == 4 and model_shape[1] == 1 and checkpoint_shape[1] == 3:
                    print(f"Adapting layer {clean_k}: 3 channels -> 1 channel")
                    new_state_dict[clean_k] = v.sum(dim=1, keepdim=True)
                else:
                    pass # Пропускаем несовпадающие
            
        self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded {len(new_state_dict)} layers from custom checkpoint.")

    def forward(self, x):
        return self.model(x)