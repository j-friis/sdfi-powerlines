import MinkowskiEngine as ME
import torch
import torch.nn as nn


class Stats(nn.Module):
    def __init__(self, kernel_size, stride, dilation, dimension):
        super().__init__()
        self.max_pool = ME.MinkowskiMaxPooling(
            kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=dimension
        )
        self.sum_pool = ME.MinkowskiSumPooling(
            kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=dimension
        )
        self.mean_pool = ME.MinkowskiAvgPooling(
            kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=dimension
        )
        self.unpool = ME.MinkowskiPoolingTranspose(
            kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=dimension
        )
        self.glob_sum_pool = ME.MinkowskiGlobalSumPooling()
        self.broadcast = ME.MinkowskiBroadcast()
        self.num_feat = 2 * dimension + 1

    def forward(self, x: ME.SparseTensor):
        xx = ME.SparseTensor(
            torch.ones(x.shape[0], 1), tensor_stride=x.tensor_stride,
            coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager,
            quantization_mode=x.quantization_mode, device=x.device,
            requires_grad=False
        )
        glob_xx = self.glob_sum_pool(xx)
        pool_xx = self.sum_pool(xx)

        density = pool_xx / self.broadcast(pool_xx, glob_xx)

        xx = ME.SparseTensor(
            x.coordinates[:, 1:].float(), tensor_stride=x.tensor_stride,
            coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager,
            quantization_mode=x.quantization_mode, device=x.device,
            requires_grad=False
        )
        center = self.mean_pool(xx)
        mean_diff = (xx - self.unpool(center))  # TODO I think this does not work as wanted
        variance = self.sum_pool(mean_diff * mean_diff) / pool_xx
        norm_center = center - center.coordinates[:, 1:].float()

        return ME.cat(density, variance, norm_center)  # 7 features


class StatsConv(Stats):
    def __init__(self, inplanes, outplanes, kernel_size, stride, dilation, dimension, bias=True):
        super().__init__(kernel_size, stride, dilation, dimension)
        self.conv = ME.MinkowskiConvolution(
            inplanes, outplanes - self.num_feat, kernel_size=kernel_size, stride=stride, dilation=dilation,
            dimension=dimension,
            bias=bias
        )

    def forward(self, x: ME.SparseTensor):
        coord_features = super().forward(x)
        x = self.conv(x)
        return ME.cat(x, coord_features)


class StatsPool(Stats):
    def __init__(self, kernel_size, stride, dimension, feature_pool=ME.MinkowskiMaxPooling):
        super().__init__(kernel_size, stride, 1, dimension)
        self.feat_pool = feature_pool(kernel_size=kernel_size, stride=stride, dimension=dimension)

    def forward(self, x: ME.SparseTensor):
        coord_features = super().forward(x)
        x = self.feat_pool(x)
        return ME.cat(x, coord_features)
