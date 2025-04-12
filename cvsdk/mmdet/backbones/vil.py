import torch
import torch.nn as nn
from mmdet.registry import MODELS
from cvsdk.mmdet.vision_lstm.vision_lstm2 import VisionLSTM2


IMG_HEIGHT, IMG_WIDTH = 512, 512
PATCH_SIZE = 16
N_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)
EMBED_DIM = 128


@MODELS.register_module()
class ViL(nn.Module):
    """Vision xLSTM (ViL) backbone."""
    def __init__(self, dim=192, input_shape=(3, 512, 512), patch_size=16, depth=6, drop_path_rate=0.0) -> None:
        """Initialize ViL backbone."""
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.input_shape = input_shape
        self.drop_path_rate = drop_path_rate
        self.vil = VisionLSTM2(
            dim=dim,  # latent dimension (192 for ViL-T)
            depth=depth,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
            patch_size=patch_size,  # patch_size (results in 64 patches for 32x32 images)
            input_shape=input_shape,  # RGB images
            drop_path_rate=drop_path_rate,  # stochastic depth parameter (disabled for ViL-T)
            mode="features" # "features" or "classifier"
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward image through ViL backbone.

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            list[torch.Tensor]: list containing tensor of shape (C, H, W) for each input image
        """
        B, C, H, W = x.shape
        print("B, C, H, W", B, C, H, W, x.shape)
        y = self.vil(x) # shape [B, H*W, C]
        print("Y", y.shape)
        y_2d = y.reshape((B, H // self.patch_size, W // self.patch_size, self.dim)).permute(0,3,1,2) # shape [B, C, H, W]
        return y_2d

