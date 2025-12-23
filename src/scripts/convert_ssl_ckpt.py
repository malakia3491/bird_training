import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Путь к last.ckpt от SSL")
    parser.add_argument("--output", type=str, required=True, help="Куда сохранить weights.pth")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}

    for k, v in state_dict.items():
        # Нам нужны только веса backbone
        if k.startswith("backbone."):
            # Убираем префикс "backbone."
            new_key = k.replace("backbone.", "") 
            # Если внутри есть еще "model." (как в PANNs wrapper), оставляем или убираем
            # В AudioBackbone (timm) у тебя self.model.
            # Поэтому ключи должны начинаться с "model."
            
            # Проверка: если в state_dict ключи типа "backbone.model.conv1...",
            # то после replace останется "model.conv1...". Это идеально для AudioBackbone.
            new_state_dict[new_key] = v

    print(f"Extracted {len(new_state_dict)} keys.")
    torch.save(new_state_dict, args.output)
    print(f"Saved backbone weights to: {args.output}")

if __name__ == "__main__":
    main()