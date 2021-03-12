import torch.nn as nn

from .backbones.resnet import ResnetDecoder, r2plus1d_18


class RODNetResnet18(nn.Module):
    def __init__(self, n_class):
        super(RODNetResnet18, self).__init__()

        self.encode = r2plus1d_18()
        self.stem = self.encode.stem
        self.layer1 = self.encode.layer1
        self.layer2 = self.encode.layer2
        self.layer3 = self.encode.layer3
        self.layer4 = self.encode.layer4
        self.decode = ResnetDecoder(n_class)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x) # [-1, 64, 16, 64, 64]
        x2 = self.layer2(x1) # [-1, 128, 8, 32, 32]
        x3 = self.layer3(x2) # [-1, 256, 4, 16, 16]
        x4 = self.layer4(x3) # [-1, 512, 2, 8, 8]
        dets = self.decode(x1, x2, x3, x4)
        return dets