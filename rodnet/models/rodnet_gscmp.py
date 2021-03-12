import torch.nn as nn

from .backbones.gscmp import GSCStack


class RODNetGSCmp(nn.Module):
    def __init__(self, n_class, stacked_num=2):
        super(RODNetGSCmp, self).__init__()
        self.stacked_hourglass = GSCStack(n_class, stacked_num=stacked_num)

    def forward(self, x):
        out = self.stacked_hourglass(x)
        return out
