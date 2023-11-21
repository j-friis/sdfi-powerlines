import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils import spectral_norm
from torch_geometric.nn import global_max_pool, global_mean_pool

from torch_points3d.core.common_modules.base_modules import *


def point_block(in_dims, out_dims, activation=None, norm=True):
    '''
    helper function to create a conv1d layer with kernel size 1, batch norm, and activation
    '''
    modules = []
    if norm:
        modules.append(nn.BatchNorm1d(in_dims))

    if activation is not None:
        modules.append(activation)

    modules.append(nn.Conv1d(in_dims, out_dims, 1))

    return nn.Sequential(*modules)


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)


def mlp(in_dims, out_dims, activation, norm=True):
    '''
    helper function to create a linear layer with batch norm and activation
    '''
    modules = []
    if norm:
        modules.append(nn.BatchNorm1d(in_dims))

    if activation is not None:
        modules.append(activation)

    modules.append(nn.Linear(in_dims, out_dims))
    return nn.Sequential(*modules)


class SelfAttention(nn.Module):
    "Self attention layer for nd."

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o


class Attention(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dims=None):
        super(Attention, self).__init__()
        if hidden_dims is None:
            hidden_dims = out_channel

        self.act = nn.ReLU(inplace=True)
        self.point_blocks = nn.Sequential(
            point_block(in_channel, hidden_dims, self.act),
        )
        self.query = nn.Sequential(
            point_block(hidden_dims, out_channel, self.act),
        )

        if out_channel != in_channel:
            self.value = point_block(in_channel, out_channel, None, norm=False)
            self.out = point_block(in_channel, out_channel, None, norm=False)
        else:
            self.value = nn.Sequential()
            self.out = nn.Sequential()

    def forward(self, x):
        n_points = x.shape[2]
        x_ = self.point_blocks(x)
        x_ = torch.max(x_, 2, keepdim=True)[0]
        x_ = self.query(x_)
        return x_.repeat_interleave(n_points, 2).sigmoid() * self.value(x) + self.out(x)


class SplitPoint(nn.Module):
    def __init__(self, in_channel):
        super(SplitPoint, self).__init__()
        self.mlp = point_block(in_channel, 1, nn.ReLU(inplace=True))

    def forward(self, x):
        bs, c, n_points = x.shape
        split = self.mlp(x).sigmoid()
        split_idx = split.argsort(2, descending=True)
        # revert_mask = torch.arange(n_points).reshape(1, 1, n_points).repeat_interleave(bs, 0)
        x = torch.cat([x, split], 1).gather(2, split_idx[:, :, :n_points // 2].repeat_interleave(c + 1, 1))

        return x  # , revert_mask


class MergePoint(nn.Module):
    def __init__(self, in_channel, skip_channel):
        super(MergePoint, self).__init__()
        self.global_feat = point_block(in_channel, skip_channel, norm=False, activation=None)

    def forward(self, x, skip):
        x = self.global_feat(x).max(2, keepdim=True)[0]
        x = torch.cat([x.repeat_interleave(skip.shape[2], 2), skip], 1)
        return x


class PointAttSeg(torch.nn.Module):
    def __init__(
            self,
            input_nc=0,
            n_classes=2,
            hidden_dims=None,
            *args,
            **kwargs
    ):
        super().__init__()

        ins = [input_nc, *hidden_dims]
        self.seg = nn.Sequential(
            point_block(ins[0], hidden_dims[0], None, False),

            Attention(hidden_dims[0], hidden_dims[0]),
            # SelfAttention(hidden_dims[0]),

            Attention(ins[1], hidden_dims[1]),
            # SelfAttention(hidden_dims[1]),

            Attention(ins[2], hidden_dims[2]),
            # SelfAttention(hidden_dims[2]),

            Attention(ins[3], hidden_dims[3]),

            Attention(ins[4], hidden_dims[4]),

            point_block(ins[5], n_classes, None),
        )
        self.n_classes = n_classes

    def forward(self, x):
        # extract global features and produce output
        out = self.seg(x)

        return out


class UPointAttSeg(nn.Module):
    def __init__(self,
            input_nc=0,
            n_classes=2,
            hidden_dims=None,
            *args,
            **kwargs):
        super(UPointAttSeg, self).__init__()
        self.n_classes = n_classes
        self.act = nn.ReLU(inplace=True)

        # global feature extractor
        self.block1 = nn.Sequential(
            Attention(input_nc, hidden_dims[0]),
            point_block(hidden_dims[0], hidden_dims[0], self.act),
        )

        self.block2 = nn.Sequential(
            SplitPoint(hidden_dims[0]),
            Attention(hidden_dims[0] + 1, hidden_dims[1]),
            point_block(hidden_dims[1], hidden_dims[1], self.act),
        )

        self.block3 = nn.Sequential(
            SplitPoint(hidden_dims[1]),
            Attention(hidden_dims[1] + 1, hidden_dims[2]),
            point_block(hidden_dims[2], hidden_dims[2], self.act),
        )

        self.block4 = nn.Sequential(
            SplitPoint(hidden_dims[2]),
            Attention(hidden_dims[2] + 1, hidden_dims[3]),
            point_block(hidden_dims[3], hidden_dims[3], self.act),
        )

        self.cross_block = nn.Sequential(
            SplitPoint(hidden_dims[3]),
            Attention(hidden_dims[3] + 1, hidden_dims[4]),
            point_block(hidden_dims[4], hidden_dims[4], self.act),
            point_block(hidden_dims[4], hidden_dims[4], self.act),
            SelfAttention(hidden_dims[4])
        )

        self.up1_merge = MergePoint(hidden_dims[4], hidden_dims[3])
        self.up1 = nn.Sequential(
            Attention(hidden_dims[3] * 2, hidden_dims[3]),
            point_block(hidden_dims[3], hidden_dims[3], self.act),
        )

        self.up2_merge = MergePoint(hidden_dims[3], hidden_dims[2])
        self.up2 = nn.Sequential(
            Attention(hidden_dims[2] * 2, hidden_dims[2]),
            point_block(hidden_dims[2], hidden_dims[2], self.act),
        )

        self.up3_merge = MergePoint(hidden_dims[2], hidden_dims[1])
        self.up3 = nn.Sequential(
            Attention(hidden_dims[1] * 2, hidden_dims[1]),
            point_block(hidden_dims[1], hidden_dims[1], self.act),
        )

        self.up4_merge = MergePoint(hidden_dims[1], hidden_dims[0])
        self.up4 = nn.Sequential(
            Attention(hidden_dims[0] * 2, hidden_dims[0]),
            point_block(hidden_dims[0], hidden_dims[0], self.act),
        )

        self.out = point_block(hidden_dims[0], n_classes, self.act)

    def forward(self, x):
        bs = x.shape[0]
        n_pts = x.shape[2]

        # extract global features and aggregate half the points per block
        x1 = self.block1(x)

        x2 = self.block2(x1)

        x3 = self.block3(x2)

        x4 = self.block4(x3)

        # cross
        cross = self.cross_block(x4)

        up1 = self.up1_merge(cross, x4)
        up1 = self.up1(up1)

        up2 = self.up2_merge(up1, x3)
        up2 = self.up2(up2)

        up3 = self.up3_merge(up2, x2)
        up3 = self.up3(up3)

        up4 = self.up4_merge(up3, x1)
        up4 = self.up4(up4)

        x = self.out(up4)
        return x