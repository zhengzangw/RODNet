import torch.nn as nn


def _make_conv(
    in_channels,
    out_channels,
    stride=(1, 1, 1),
    kernel_size=(9, 5, 5),
    padding=(4, 2, 2),
    bn=False,
    relu=False,
):
    conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    layer = [conv]
    if bn:
        layer.append(nn.BatchNorm3d(out_channels))
    if relu:
        layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def _make_pool(kernel_size=(2, 2, 2), padding=(0, 0, 0), stride=(2, 2, 2)):
    return nn.MaxPool3d(kernel_size=kernel_size, padding=padding, stride=stride)


C = 64


class GSCStack(nn.Module):
    def __init__(self, n_class, stacked_num=1, in_channels=2):
        super(GSCStack, self).__init__()
        self.stacked_num = stacked_num
        self.conv1 = _make_conv(in_channels, C, relu=True)

        self.stacks = []
        for i in range(stacked_num):
            self.stacks.append(
                nn.ModuleList(
                    [
                        GSCEncode(),
                        GSCDecode(),
                        _make_conv(C, n_class),
                        _make_conv(n_class, C),
                    ]
                )
            )
        self.stacks = nn.ModuleList(self.stacks)

    def forward(self, x):
        x = self.conv1(x)

        out = []
        for i in range(self.stacked_num):
            x, x1, x2, x3 = self.stacks[i][0](x)
            x = self.stacks[i][1](x, x1, x2, x3)
            confmap = self.stacks[i][2](x)
            out.append(confmap)
            if i < self.stacked_num - 1:
                confmap_ = self.stacks[i][3](confmap)
                x = x + confmap_

        return out


class GSCEncode(nn.Module):
    def __init__(self):
        super(GSCEncode, self).__init__()
        self.conv1 = _make_conv(C, C)
        self.pool1 = _make_pool()
        self.conv2 = _make_conv(C, C)
        self.pool2 = _make_pool()
        self.conv3 = _make_conv(C, C)
        self.pool3 = _make_pool(stride=(1, 2, 2), kernel_size=(1, 2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x1 = x  # [Bx64xWx128x128]
        x = self.pool1(x)
        x = self.conv2(x)
        x2 = x  # [Bx64xW/2x64x64]
        x = self.pool2(x)
        x = self.conv3(x)
        x3 = x  # [Bx128xW/4x32x32]
        x = self.pool3(x)
        # [Bx256xW/4x16x16]

        return x, x1, x2, x3


class GSCDecode(nn.Module):
    def __init__(self):
        super(GSCDecode, self).__init__()
        self.upsample3 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        self.conv3 = _make_conv(C, C)
        self.conv3c = _make_conv(C, C)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear")
        self.conv2 = _make_conv(C, C)
        self.conv2c = _make_conv(C, C)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear")
        self.conv1 = _make_conv(C, C)
        self.conv1c = _make_conv(C, C)

    def forward(self, x, x1, x2, x3):
        # [Bx256xW/4x16x16]
        x = self.upsample3(x)
        x3 = self.conv3(x3)
        x = self.conv3c(x + x3)
        # [Bx128xW/4x32x32]
        x = self.upsample2(x)
        x2 = self.conv2(x2)
        x = self.conv2c(x + x2)
        # [Bx64xW/2x64x64]
        x = self.upsample1(x)
        x1 = self.conv1(x1)
        x = self.conv1c(x + x1)
        return x
