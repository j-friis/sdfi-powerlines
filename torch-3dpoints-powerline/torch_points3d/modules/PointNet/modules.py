import torch
from torch_geometric.nn import global_max_pool, global_mean_pool

from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.core.common_modules.spatial_transform import BaseLinearTransformSTNkD
from torch_points3d.models.base_model import BaseInternalLossModule


class MiniPointNet(torch.nn.Module):
    def __init__(self, local_nn, global_nn, aggr="max", return_local_out=False):
        super().__init__()

        self._local_nn = MLP(local_nn)
        self._global_nn = MLP(global_nn) if global_nn else None
        self._aggr = aggr
        self.g_pool = global_max_pool if aggr == "max" else global_mean_pool
        self.return_local_out = return_local_out

    def forward(self, x, batch):
        y = x = self._local_nn(x)  # [num_points, in_dim] -> [num_points, local_out_nn]
        if batch is not None:
            x = self.g_pool(x, batch)  # [num_points, local_out_nn] -> [local_out_nn]
        else:
            x = x.max(1)[0] if self._aggr == 'max' else x.mean(1)
        if self._global_nn:
            x = self._global_nn(x)  # [local_out_nn] -> [global_out_nn]
        if self.return_local_out:
            return x, y
        return x

    def forward_embedding(self, pos, batch):
        global_feat, local_feat = self.forward(pos, batch)
        indices = batch.unsqueeze(-1).repeat((1, global_feat.shape[-1]))
        gathered_global_feat = torch.gather(global_feat, 0, indices)
        x = torch.cat([local_feat, gathered_global_feat], -1)
        return x


class PointNetSTN3D(BaseLinearTransformSTNkD):
    def __init__(self, local_nn=[3, 64, 128, 1024], global_nn=[1024, 512, 256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1], 3, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)


class PointNetSTNkD(BaseLinearTransformSTNkD, BaseInternalLossModule):
    def __init__(self, k=64, local_nn=[64, 64, 128, 1024], global_nn=[1024, 512, 256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1], k, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)

    def get_internal_losses(self):
        return {"orthogonal_regularization_loss": self.get_orthogonal_regularization_loss()}


class PointNetSeg(torch.nn.Module):
    def __init__(
            self,
            input_nc=3,
            input_stn_local_nn=[64, 128, 1024],
            input_stn_global_nn=[1024, 512, 256],
            local_nn_1=[64, 64],
            feat_stn_k=64,
            feat_stn_local_nn=[64, 64, 128, 1024],
            feat_stn_global_nn=[1024, 512, 256],
            local_nn_2=[64, 64, 128, 1024],
            seg_nn=[1088, 512, 256, 128, 4],
            batch_size=1,
            *args,
            **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size

        self.input_stn = PointNetSTN3D([input_nc] + input_stn_local_nn, input_stn_global_nn, batch_size)
        self.local_nn_1 = MLP([input_nc] + local_nn_1)
        self.feat_stn = PointNetSTNkD(feat_stn_k, feat_stn_local_nn, feat_stn_global_nn, batch_size)
        self.local_nn_2 = MLP(local_nn_2)
        self.seg_nn = MLP(seg_nn)

        self._use_scatter_pooling = True

    def set_scatter_pooling(self, use_scatter_pooling):
        self._use_scatter_pooling = use_scatter_pooling

    def func_global_max_pooling(self, x, batch):
        if self._use_scatter_pooling:
            return global_max_pool(x, batch)
        else:
            return x.max(1)[0]

    def forward(self, x, batch):

        # apply pointnet classification network to get per-point
        # features and global feature
        x = self.input_stn(x, batch)
        x = self.local_nn_1(x)
        x_feat_trans = self.feat_stn(x, batch)
        x3 = self.local_nn_2(x_feat_trans)

        global_feature = self.func_global_max_pooling(x3, batch)
        # concat per-point and global feature and regress to get
        # per-point scores
        if x_feat_trans.dim() == 2:
            feat_concat = torch.cat([x_feat_trans, global_feature[batch]], dim=1)
        else:
            feat_concat = torch.cat([x_feat_trans,
                                     global_feature.unsqueeze(1).repeat((1, x_feat_trans.shape[1], 1))], dim=-1)
        out = self.seg_nn(feat_concat)

        return out


class PointNetReg(torch.nn.Module):
    def __init__(
            self,
            input_nc=3,
            input_stn_local_nn=[64, 128, 1024],
            input_stn_global_nn=[1024, 512, 256],
            local_nn_1=[64, 64],
            feat_stn_k=64,
            feat_stn_local_nn=[64, 64, 128, 1024],
            feat_stn_global_nn=[1024, 512, 256],
            local_nn_2=[64, 64, 128, 1024],
            reg_nn=[1088, 512, 256, 128, 4],
            dropout=0.0,
            global_pool="max",
            batch_size=1,
            *args,
            **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size

        self.input_stn = PointNetSTN3D([input_nc] + input_stn_local_nn, input_stn_global_nn, batch_size)
        self.local_nn_1 = MLP([input_nc] + local_nn_1)
        self.feat_stn = PointNetSTNkD(feat_stn_k, feat_stn_local_nn, feat_stn_global_nn, batch_size)
        self.local_nn_2 = MLP(local_nn_2)
        self.reg_nn = MLP(reg_nn[:-1])
        if dropout > 0:
            self.reg_nn = nn.Sequential(self.reg_nn, nn.Dropout(dropout))

        self.out = nn.Linear(reg_nn[-2], reg_nn[-1])

        if global_pool == "max":
            self.func_global_pooling = self.func_global_max_pooling
        elif global_pool in ["avg", "mean"]:
            self.func_global_pooling = self.func_global_max_pooling
        elif global_pool in ["maxmean", "meanmax"]:
            self.dim_gpool = local_nn_2[-1]
            self.func_global_pooling = self.func_global_maxmean_pooling
        else:
            raise Exception(f"This global pooling is not support: {global_pool}")

        self._use_scatter_pooling = True

    def set_scatter_pooling(self, use_scatter_pooling):
        self._use_scatter_pooling = use_scatter_pooling

    def func_global_maxmean_pooling(self, x, batch):
        if self._use_scatter_pooling:
            gmax = global_max_pool(x[:, :self.dim_gpool//2], batch)
            gmean = global_mean_pool(x[:, self.dim_gpool//2:], batch)
            return torch.cat([gmax, gmean], dim=1)
        else:
            return torch.cat([x.max(1)[0], x.mean(1)], dim=1)

    def func_global_mean_pooling(self, x, batch):
        if self._use_scatter_pooling:
            return global_mean_pool(x, batch)
        else:
            return x.mean(1)

    def func_global_max_pooling(self, x, batch):
        if self._use_scatter_pooling:
            return global_max_pool(x, batch)
        else:
            return x.max(1)[0]

    def forward(self, x, batch):

        # apply pointnet classification network to get per-point
        # features and global feature
        x = self.input_stn(x, batch)
        x = self.local_nn_1(x)
        x_feat_trans = self.feat_stn(x, batch)
        x3 = self.local_nn_2(x_feat_trans)

        global_feature = self.func_global_max_pooling(x3, batch)
        out = self.reg_nn(global_feature)

        out = self.out(out)

        return out
