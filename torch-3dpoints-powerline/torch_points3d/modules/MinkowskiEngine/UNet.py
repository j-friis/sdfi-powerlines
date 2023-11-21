import MinkowskiEngine as ME
import torch.nn as nn
from MinkowskiEngine.MinkowskiOps import cat

from .SENet import ResNetBase
from .resnet_block import BasicBlock, Bottleneck
from .senet_block import SEBasicBlock, SEBottleneck


class LongSkipUpscale(nn.Module):
    def __init__(self, inplanes, D):
        super(LongSkipUpscale, self).__init__()
        self.upscale = ME.MinkowskiConvolutionTranspose(
            in_channels=inplanes, out_channels=inplanes // 2,
            kernel_size=[2] * D, stride=[2] * D, dimension=D
        )

    def forward(self, x):
        x, cross_x = x
        x = self.upscale(x)
        return cat(x, cross_x)


class ResUNetBase(ResNetBase):
    UP_PLANES = None
    UP_LAYERS = None

    def __init__(self, in_channels, out_channels, activation="relu", D=3, first_stride=1, dropout=0.0,
                 global_pool="mean", statpooling=False, bias=True,                  **kwargs):
        super().__init__(in_channels, out_channels, activation, D, first_stride, dropout,
                         global_pool, statpooling, bias, **kwargs)
        if self.UP_PLANES is None:
            self.UP_PLANES = self.PLANES[-2::-1] + (self.INIT_DIM,)

        if self.UP_LAYERS is None:
            self.UP_LAYERS = self.LAYERS[-2::-1] + (1,)

        del self.final
        del self.glob_avg

        up_blocks = []
        for cross_dim, planes, layers in zip(self.cross_dims, self.UP_PLANES, self.UP_LAYERS):
            up_blocks.append(self._make_up_layer(self.BLOCK, cross_dim, planes, layers))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = ME.MinkowskiConvolution(self.UP_PLANES[-1] * self.BLOCK.expansion, out_channels, 1, dimension=D)

    def _make_up_layer(self, block, cross_plane, planes, blocks, dilation=1):

        layers = [LongSkipUpscale(self.inplanes, self.D)]
        self.inplanes //= 2
        downsample = None
        if self.inplanes + cross_plane != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes + cross_plane, planes * block.expansion, kernel_size=1, dimension=self.D,
                    dilation=1, bias=self.bias,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )

        layers.append(
            block(
                self.inplanes + cross_plane, planes, self.act_fn, dilation=dilation,
                dimension=self.D, bias=self.bias, downsample=downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, self.act_fn, stride=1,
                dilation=dilation, dimension=self.D, bias=self.bias,
            ))

        return nn.Sequential(*layers)

    def forward(self, in_):
        down = []
        x = self.conv1(in_)
        x = self.bn1(x)
        x = self.act_fn(x)
        down.append(x)

        x = self.pool(x)
        for block in self.blocks:
            x = block(x)
            down.append(x)

        down = down[-2::-1]

        for block, cross_x in zip(self.up_blocks, down):
            x = block((x, cross_x))

        return self.final(x)


class ResUNet14_(ResUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)
    STRIDES = (1, 2, 2, 2)


class ResUNet18_(ResUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)
    STRIDES = (1, 2, 2, 2)


class ResUNet34_(ResUNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class ResUNet50_(ResUNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class ResUNet101_(ResUNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)
    STRIDES = (1, 2, 2, 2)

class ResUNet14_2UP(ResUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]


class ResUNet18_2UP(ResUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]


class ResUNet34_2UP(ResUNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]


class ResUNet50_2UP(ResUNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]


class ResUNet101_2UP(ResUNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]

class SEUNet14(ResUNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (1, 1, 1, 1)
    STRIDES = (1, 2, 2, 2)


class SEUNet18(ResUNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (2, 2, 2, 2)
    STRIDES = (1, 2, 2, 2)


class SEUNet34(ResUNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class SEUNet50(ResUNetBase):
    BLOCK = SEBottleneck
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class SEUNet101(ResUNetBase):
    BLOCK = SEBottleneck
    LAYERS = (3, 4, 23, 3)
    STRIDES = (1, 2, 2, 2)

class SEUNet14_2UP(ResUNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (1, 1, 1, 1)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]


class SEUNet18_2UP(ResUNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (2, 2, 2, 2)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]


class SEUNet34_2UP(ResUNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]


class SEUNet50_2UP(ResUNetBase):
    BLOCK = SEBottleneck
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]


class SEUNet101_2UP(ResUNetBase):
    BLOCK = SEBottleneck
    LAYERS = (3, 4, 23, 3)
    STRIDES = (1, 2, 2, 2)
    UP_PLANES = [256, 128, 96, 96]
    UP_LAYERS = [2, 2, 2, 2]