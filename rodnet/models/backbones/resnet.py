import torch.nn as nn

CHANNELS = 2


class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv3DNoTemporal(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """

    def __init__(self, n_channel=2):
        super(BasicStem, self).__init__(
            nn.Conv3d(
                n_channel,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self, n_channel=2):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(
                n_channel,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class R2Plus1dStem_b(nn.Sequential):
    def __init__(self, n_channel=2):
        super(R2Plus1dStem_b, self).__init__(
            nn.Conv3d(
                n_channel,
                45,
                kernel_size=(9, 1, 1),
                stride=(1, 1, 1),
                padding=(4, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(1, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class VideoResNet(nn.Module):
    def __init__(
        self,
        block,
        conv_makers,
        layers,
        stem,
        num_classes=400,
        n_channel=2,
        zero_init_residual=False,
    ):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem(n_channel)

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=ds_stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, **kwargs):
    model = VideoResNet(**kwargs)

    return model


def r3d_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet(
        "r3d_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs
    )


def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet(
        "mc3_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs
    )


def r2plus1d_18(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _video_resnet(
        "r2plus1d_18",
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        **kwargs
    )


def r2plus1d_18_b(pretrained=False, progress=True, **kwargs):
    return _video_resnet(
        "r2plus1d_18_b",
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem_b,
        **kwargs
    )


def _make_conv(
    in_channels,
    out_channels,
    stride=(1, 1, 1),
    kernel_size=(3, 7, 7),
    padding=(1, 3, 3),
    raw=False,
):
    conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    return nn.Sequential(conv, nn.BatchNorm3d(out_channels), nn.ReLU(inplace=False),)


class ResnetDecoder_b(nn.Module):
    def __init__(self, num_class):
        super(ResnetDecoder_b, self).__init__()
        self.conv_x3 = _make_conv(256, 512, stride=(2, 2, 2))
        self.conv_x2 = _make_conv(128, 256, stride=(2, 2, 2))
        self.conv_x1 = _make_conv(64, 128, stride=(2, 2, 2))

        self.prelu = nn.PReLU()

        self.convt4 = nn.ConvTranspose3d(
            in_channels=512,
            out_channels=256,
            kernel_size=(4, 6, 6),
            stride=(2, 2, 2),
            padding=(1, 2, 2),
        )
        self.convt3 = nn.ConvTranspose3d(
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
        self.convt1 = nn.ConvTranspose3d(
            in_channels=64,
            out_channels=num_class,
            kernel_size=(3, 6, 6),
            stride=(1, 2, 2),
            padding=(1, 2, 2),
        )

        self._initialize_weights()

    def forward(self, x1, x2, x3, x4):
        # x1 [-1, 64, 16, 64, 64]
        # x2 [-1, 128, 8, 32, 32]
        # x3 [-1, 256, 4, 16, 16]
        # x4 [-1, 512, 2, 8, 8]

        x3 = self.conv_x3(x3)  # [-1, 512, 2, 8, 8]
        x2 = self.conv_x2(x2)  # [-1, 256, 4, 16, 16]
        x1 = self.conv_x1(x1)  # [-1, 128, 8, 32, 32]

        x = x4
        x = self.prelu(self.convt4(x + x3))
        x = self.prelu(self.convt3(x + x2))
        x = self.prelu(self.convt2(x + x1))
        x = self.convt1(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ResnetDecoder(nn.Module):
    def __init__(self, num_class):
        super(ResnetDecoder, self).__init__()
        self.upsample4 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=False
        )
        self.upsample3 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=False
        )
        self.upsample2 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=False
        )
        self.upsample1 = nn.Upsample(
            scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
        )

        self.conv4 = _make_conv(512, 256)
        self.conv3 = _make_conv(256, 128)
        self.conv2 = _make_conv(128, 64)
        self.conv1 = _make_conv(64, num_class)

        self.conv_x3 = _make_conv(256, 512)
        self.conv_x2 = _make_conv(128, 256)
        self.conv_x1 = _make_conv(64, 128)

        self._initialize_weights()

    def forward(self, x1, x2, x3, x4):
        # x1 [-1, 64, 16, 64, 64]
        # x2 [-1, 128, 8, 32, 32]
        # x3 [-1, 256, 4, 16, 16]
        # x4 [-1, 512, 2, 8, 8]
        x = self.upsample4(x4)  # [-1, 512, 4, 16, 16]
        x3 = self.conv_x3(x3)  # [-1, 512, 4, 16, 16]
        x = self.conv4(x + x3)  # [-1, 256, 4, 16, 16]

        x = self.upsample3(x)  # [-1, 256, 8, 32, 32]
        x2 = self.conv_x2(x2)  # [-1, 256, 8, 32, 32]
        x = self.conv3(x + x2)  # [-1, 128, 8, 32, 32]

        x = self.upsample2(x)  # [-1, 128, 16, 64, 64]
        x1 = self.conv_x1(x1)
        x = self.conv2(x + x1)  # [-1, 64, 16, 64, 64]

        x = self.upsample1(x)  # [-1, 64, 16, 128, 128]
        x = self.conv1(x)  # [-1, n_cls, 16, 128, 128]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Resnet3Decoder(nn.Module):
    def __init__(self, num_class):
        super(ResnetDecoder, self).__init__()
        self.upsample4 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=False
        )
        self.upsample3 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=False
        )
        self.upsample2 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=False
        )
        self.upsample1 = nn.Upsample(
            scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
        )

        self.conv4 = _make_conv(512, 256)
        self.conv3 = _make_conv(256, 128)
        self.conv2 = _make_conv(128, 64)
        self.conv1 = _make_conv(64, num_class)

        self.conv_x3 = _make_conv(256, 512)
        self.conv_x2 = _make_conv(128, 256)
        self.conv_x1 = _make_conv(64, 128)

        self._initialize_weights()

    def forward(self, x1, x2, x3, x4):
        # x1 [-1, 64, 16, 64, 64]
        # x2 [-1, 128, 8, 32, 32]
        # x3 [-1, 256, 4, 16, 16]
        # x4 [-1, 512, 2, 8, 8]
        x = self.upsample4(x4)  # [-1, 512, 4, 16, 16]
        x3 = self.conv_x3(x3)  # [-1, 512, 4, 16, 16]
        x = self.conv4(x + x3)  # [-1, 256, 4, 16, 16]

        x = self.upsample3(x)  # [-1, 256, 8, 32, 32]
        x2 = self.conv_x2(x2)  # [-1, 256, 8, 32, 32]
        x = self.conv3(x + x2)  # [-1, 128, 8, 32, 32]

        x = self.upsample2(x)  # [-1, 128, 16, 64, 64]
        x1 = self.conv_x1(x1)
        x = self.conv2(x + x1)  # [-1, 64, 16, 64, 64]

        x = self.upsample1(x)  # [-1, 64, 16, 128, 128]
        x = self.conv1(x)  # [-1, n_cls, 16, 128, 128]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

