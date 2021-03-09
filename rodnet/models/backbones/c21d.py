import torch.nn as nn


def _make_conv(
    in_channels, out_channels, stride, kernel_size=(9, 5, 5), padding=(4, 2, 2)
):
    mid_channels = (
        in_channels * out_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
    ) // (in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels)
    # mid_channels = out_channels
    return nn.Sequential(
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
        ),
        nn.BatchNorm3d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=(1, *kernel_size[1:]),
            stride=(1, *stride[1:]),
            padding=(0, *padding[1:]),
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class RODEncode(nn.Module):
    def __init__(self, in_channels=2):
        super(RODEncode, self).__init__()
        self.conv1a = _make_conv(in_channels, 64, (1, 1, 1))
        self.conv1b = _make_conv(64, 64, (2, 2, 2))
        self.conv2a = _make_conv(64, 128, (1, 1, 1))
        self.conv2b = _make_conv(128, 128, (2, 2, 2))
        self.conv3a = _make_conv(128, 256, (1, 1, 1))
        self.conv3b = _make_conv(256, 256, (1, 2, 2))

    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        return x


class RODDecode(nn.Module):
    def __init__(self, n_class):
        super(RODDecode, self).__init__()
        self.convt1 = nn.ConvTranspose3d(
            in_channels=256,
            out_channels=128,
            kernel_size=(4, 6, 6),
            stride=(2, 2, 2),
            padding=(1, 2, 2),
        )
        self.convt2 = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=64,
            kernel_size=(4, 6, 6),
            stride=(2, 2, 2),
            padding=(1, 2, 2),
        )
        self.convt3 = nn.ConvTranspose3d(
            in_channels=64,
            out_channels=n_class,
            kernel_size=(3, 6, 6),
            stride=(1, 2, 2),
            padding=(1, 2, 2),
        )
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.prelu(self.convt1(x))
        x = self.prelu(self.convt2(x))
        x = self.convt3(x)
        return x

