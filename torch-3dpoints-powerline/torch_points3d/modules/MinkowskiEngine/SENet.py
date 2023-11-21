from functools import partial

import MinkowskiEngine as ME
import torch
import torch.nn as nn
from MinkowskiEngine import MinkowskiNonlinearity as NL
from MinkowskiEngine import MinkowskiNormalization as N

from .resnet_block import BasicBlock, Bottleneck
from .senet_block import SEBasicBlock, SEBottleneck
from .statspool import StatsConv, StatsPool

ACTIVATIONS = {
    "relu": partial(NL.MinkowskiReLU, inplace=True),
    "celu": partial(NL.MinkowskiCELU, inplace=True),
    "silu": partial(NL.MinkowskiSiLU, inplace=True),
    "swish": partial(NL.MinkowskiSiLU, inplace=True),
    "gelu": NL.MinkowskiGELU,
    "elu": partial(NL.MinkowskiELU, inplace=True, alpha=0.54),
}

NORMLAYERS = {
    "bn": N.MinkowskiBatchNorm,
    "in": N.MinkowskiInstanceNorm,
}

# class GlobalMaxMeanPooling(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.gmean = ME.MinkowskiGlobalAvgPooling()
#         self.gmax = ME.MinkowskiGlobalMaxPooling()
#
#     def forward(self, x):
#         return ME.cat([self.gmean(x), self.gmax(x)], dim=1)


GLOBAL_POOL = {
    "max": ME.MinkowskiGlobalMaxPooling,
    "mean": ME.MinkowskiGlobalAvgPooling,
}


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, activation="relu", D=3, first_stride=2, dropout=0.0,
                 global_pool="mean", statpooling=False, bias=True, **kwargs):
        nn.Module.__init__(self)
        self.D = D
        self.statpooling = statpooling
        self.bias = bias
        self.cross_dims = []
        assert self.BLOCK is not None, "BLOCK is not defined"
        assert self.PLANES is not None, "PLANES is not defined"
        assert self.STRIDES is not None, "STRIDES is not defined"

        self.act_fn = ACTIVATIONS[activation]()

        self.inplanes = self.INIT_DIM
        first_out_planes = self.inplanes - (7 if statpooling else 0)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, first_out_planes, kernel_size=7, stride=first_stride, dimension=D,
            bias=bias
        )

        self.bn1 = ME.MinkowskiBatchNorm(first_out_planes)
        self.cross_dims.append(first_out_planes)

        if statpooling:
            self.pool = StatsPool(kernel_size=2, stride=2, dimension=D)
        else:
            self.pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)

        self.blocks = []
        for planes, layers, stride in zip(self.PLANES, self.LAYERS, self.STRIDES):
            self.blocks.append(
                self._make_layer(self.BLOCK, planes, layers, stride=stride)
            )
            self.cross_dims.append(self.inplanes)
        self.blocks = nn.Sequential(*self.blocks)

        self.glob_avg = GLOBAL_POOL[global_pool]()  # dimension=D)
        if dropout > 0:
            self.glob_avg = nn.Sequential(
                ME.MinkowskiDropout(dropout),
                self.glob_avg,
            )

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

        self.cross_dims = self.cross_dims[-2::-1]

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, dimension=self.D,
                    dilation=1, bias=self.bias,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes, planes, self.act_fn, stride=stride, dilation=dilation,
                downsample=downsample, dimension=self.D, statpooling=self.statpooling, bias=self.bias,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, self.act_fn, stride=1,
                dilation=dilation, dimension=self.D, bias=self.bias,
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.pool(x)

        x = self.blocks(x)

        x = self.glob_avg(x)
        return self.final(x)


class ResNet14_(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)
    STRIDES = (1, 2, 2, 2)


class ResNet18_(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)
    STRIDES = (1, 2, 2, 2)


class ResNet34_(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class ResNet50_(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class ResNet101_(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)
    STRIDES = (1, 2, 2, 2)


class SENet14(ResNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (1, 1, 1, 1)
    STRIDES = (1, 2, 2, 2)


class SENet18(ResNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (2, 2, 2, 2)
    STRIDES = (1, 2, 2, 2)


class SENet34(ResNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class SENet50(ResNetBase):
    BLOCK = SEBottleneck
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class SENet101(ResNetBase):
    BLOCK = SEBottleneck
    LAYERS = (3, 4, 23, 3)
    STRIDES = (1, 2, 2, 2)
