import torch.nn as nn

from .backbones.gsc import GSCStack


class RODNetGSC(nn.Module):
    def __init__(self, n_class, stacked_num=2):
        super(RODNetGSC, self).__init__()
        self.stacked_hourglass = GSCStack(n_class, stacked_num=stacked_num)

    def forward(self, x):
        out = self.stacked_hourglass(x)
        return out
