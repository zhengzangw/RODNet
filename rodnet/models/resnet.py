import torch.nn as nn

from .backbones.resnet import r2plus1d_18


class Resnet18(nn.Module):
    def __init__(self, n_class):
        super(Resnet18, self).__init__()

        self.resnet = r2plus1d_18(num_classes=n_class)

    def forward(self, x):
        x = self.resnet(x)
        return x

