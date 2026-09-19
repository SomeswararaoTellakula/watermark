
import torch.nn as nn
import numpy as np
from noise_layers import *


class Random_Noise(nn.Module):

    def __init__(self, layers):
        super(Random_Noise, self).__init__()
        self.noise_layers = nn.ModuleList()
        for layer in layers:
            if isinstance(layer, str):
                cls = eval(layer)
                mod = cls() if isinstance(cls, type) else cls
            elif isinstance(layer, type):
                mod = layer()
            else:
                mod = layer
            if mod is not None:
                self.noise_layers.append(mod)

    def forward(self, image_cover):
        image = image_cover[0]
        cover_image = image_cover[1]
        forward_image = image.clone().detach()
        forward_cover_image = cover_image.clone().detach()
        forward_image_cover = [forward_image, forward_cover_image]
        if len(self.noise_layers) == 0:
            return image
        idx = np.random.choice(len(self.noise_layers))
        noise_layer = self.noise_layers[idx]
        noised_image = noise_layer(forward_image_cover)
        noised_image_gap = noised_image - forward_image
        return image + noised_image_gap

