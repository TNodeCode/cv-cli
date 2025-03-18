import torch

model = torch.hub.load('facebookresearch/xcit:main', "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", pretrained=False)

print(model)