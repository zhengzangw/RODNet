import torch.nn as nn

from .backbones.c21d import RODEncode, RODDecode


class RODNetC21D(nn.Module):
    def __init__(self, n_class):
        super(RODNetC21D, self).__init__()

        self.c3d_encode = RODEncode()
        self.c3d_decode = RODDecode(n_class)

    def forward(self, x):
        x = self.c3d_encode(x)
        dets = self.c3d_decode(x)
        return dets
