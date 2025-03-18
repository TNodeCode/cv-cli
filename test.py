import torch
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
print(vits16)