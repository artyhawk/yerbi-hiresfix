import torch
import torch.nn as nn
import torch.nn.functional as F

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(UpscaleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.leaky_relu(x)
        return x

class Yerbi(nn.Module):
    def __init__(self, input_channels=4, feature_channels=64, num_upscale_blocks=2):
        super(Yerbi, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, feature_channels, kernel_size=3, stride=1, padding=1)
        
        # Upscaling blocks
        self.upscale_blocks = nn.Sequential(
            *[UpscaleBlock(feature_channels, feature_channels) for _ in range(num_upscale_blocks)]
        )
        
        # Final output block
        self.final_conv = nn.Conv2d(feature_channels, input_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.leaky_relu(self.initial_conv(x))
        x = self.upscale_blocks(x)
        x = self.final_conv(x)
        return x
