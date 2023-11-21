import torch.utils.data
from torch_geometric.nn import knn_graph, global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax

from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.core.common_modules.gathering import gather


def mlp(in_dims, out_dims, activation):
    '''
    helper function to create a linear layer with batch norm and activation
    '''
    modules = [
        nn.Linear(in_dims, out_dims, bias=False),
        FastBatchNorm1d(out_dims)
    ]
    if activation is not None:
        modules.append(activation)
    return nn.Sequential(*modules)


def absolute_feature_cat(pos, x):
    return torch.cat([pos, x], 1)


def relative_feature_cat(pos, x):
    return x

# TODO
def absolute_z_only_cat(pos, x):
    return torch.cat([pos[:, [-1]], x], 1)


class IgnoreSequential(nn.Sequential):
    def forward(self, input, y):
        for module in self:
            input = module(input)
        return input


class Softmax(nn.Module):
    def forward(self, x, batch):
        return softmax(x, batch)

class Attention(nn.Module):
    '''
    similar to channel attention in
    CBAM https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
    '''

    def __init__(self, in_dim, reduction_ratio=16):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(in_dim, in_dim//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//reduction_ratio, in_dim)
        )

    def forward(self, x, batch):
        q = self.query(x)
        q = global_max_pool(q, batch) + global_mean_pool(q, batch)
        return q.sigmoid()[batch] * x

class LocalAtt(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.value = mlp(in_dim, out_dim, act)

    def forward(self, logit, x):
        return self.value(x) * logit.sigmoid()


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def noop(): return nn.Sequential()

class NeighborPointInteractX(nn.Module):
    def __init__(self, in_dim: int, n_coord: int, out_dim: int, neighbor_pos: str, agg: str):
        super().__init__()
        in_dim_xi = in_dim + (0 if neighbor_pos == "relative" else n_coord)
        self.local_block_xi = nn.Linear(in_dim_xi, out_dim)
        self.local_block_xn = nn.Linear(in_dim + n_coord, out_dim)
        if neighbor_pos == "relative":
            self.feature_cat = relative_feature_cat
        else:
            self.feature_cat = absolute_feature_cat

        self.is_avg = False

        if agg == "max":
            self.pool = torch.amax
        elif agg == "avg":
            self.pool = torch.sum
            self.is_avg = True
        elif agg == "sum":
            self.pool = torch.sum

    def forward(self, query_pos, key_pos, idx_neighbors, query_x, key_x):
        xi = self.local_block_xi(self.feature_cat(query_pos, query_x))
        mask = (idx_neighbors!=-1).unsqueeze(2)
        neigh_in = torch.cat([
            (gather(key_pos, idx_neighbors) - query_pos.unsqueeze(1)),
            gather(key_x, idx_neighbors)
        ], 2)
        x_neigh = self.local_block_xn(neigh_in)
        x = (x_neigh + xi.unsqueeze(1)) * mask
        x = self.pool(x, 1)  # reduce to number of points
        if self.is_avg:
            x = x/mask.sum(1)
        return x

class NeighborPointInteract(nn.Module):
    def __init__(self, in_dim: int, n_coord: int, out_dim: int, neighbor_pos: str):
        super().__init__()
        in_dim_xi = in_dim + (0 if neighbor_pos == "relative" else n_coord)
        self.local_block_xi = nn.Linear(in_dim_xi, out_dim)
        self.local_block_xn = nn.Linear(in_dim + n_coord, out_dim)
        if neighbor_pos == "relative":
            self.feature_cat = relative_feature_cat
        else:
            self.feature_cat = absolute_feature_cat

    def forward(self, pos, x, neighbors, neighbor_batch):
        xi = self.local_block_xi(self.feature_cat(pos, x))

        neigh_in = torch.cat([pos[neighbors] - pos[neighbor_batch], x[neighbors]], 1)
        x_neigh = self.local_block_xn(neigh_in)

        return x_neigh + xi[neighbor_batch]

def get_gpool(agg):
    if agg == "max":
        return global_max_pool
    elif agg == "avg":
        return global_mean_pool
    elif agg == "sum":
        return global_add_pool
    # elif agg in ["mlp", "linear"]:
    #     agg_act = None if agg == "linear" else self.act
    #     return IgnoreSequential(
    #         Reshape(-1, k, out_dim),
    #         nn.Flatten(1),
    #         mlp(k * out_dim, out_dim, agg_act)
    else:
        raise NotImplementedError(f"Aggregation '{agg}' is not implemented.")

class PointInteract(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, agg: str,
                 n_coord: int, k: int, local_att: bool, global_att: bool, pre_act: bool, neighbor_pos: str):
        super(PointInteract, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.pre_act = nn.Sequential(FastBatchNorm1d(in_dim), self.act) if pre_act else noop()

        self.neighbor_interact = NeighborPointInteract(in_dim, n_coord, out_dim, neighbor_pos)
        # self.neighbor_interact_xy = NeighborPointInteract(in_dim, n_coord, out_dim, neighbor_pos)
        self.local_att = Attention(out_dim) if local_att else IgnoreSequential()

        self.g_pool = get_gpool(agg)

        self.global_att = Attention(out_dim) if global_att else IgnoreSequential()

        # self.skip = noop() if in_dim == out_dim else mlp(in_dim, out_dim, None)

    def forward(self, pos, x, batch, neighbors, neighbors_batch):

        x = self.pre_act(x)

        neigh_x = self.neighbor_interact(pos, x, neighbors, neighbors_batch)
        neigh_x = self.local_att(neigh_x, neighbors_batch)
        x = self.g_pool(neigh_x, neighbors_batch)  # reduce to number of points

        x = self.global_att(x, batch)

        return x


class PointNetSharpSeg(torch.nn.Module):
    def __init__(
            self,
            info_channel,
            local_nn,  # list of dims in local context (per layer)
            out_nn,
            k=20,
            agg="max",
            neighbor_pos="relative",
            local_att=False,
            global_att=False,
            *args,
            **kwargs
    ):
        super(PointNetSharpSeg, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.k = k
        n_coord = 3
        self.is_absolute = neighbor_pos == "absolute"
        if local_nn is None:
            raise Exception("need to specify local network sizes")

        n_layer = len(local_nn)
        in_feat = [info_channel, *local_nn]
        self.point_interact = []
        self.glob_att = []
        for i in range(n_layer):
            self.point_interact.append(
                PointInteract(
                    in_feat[i], local_nn[i], agg=agg, n_coord=n_coord, k=k,
                    local_att=local_att, global_att=global_att, neighbor_pos=neighbor_pos, pre_act=i != 0,
                )
            )

        self.point_interact = nn.ModuleList(self.point_interact)

        # segmentation mlp part
        in_dim = in_feat[-1]
        if self.is_absolute:  # if absolute positions are required
            in_dim += n_coord
            self.feature_cat = absolute_feature_cat
        else:
            self.feature_cat = relative_feature_cat
        self.final_seg = []
        if len(out_nn) > 1:
            for out_dim in list(out_nn)[:-1]:
                self.final_seg.append(mlp(in_dim, out_dim, self.act))
                in_dim = out_dim
        self.final_seg.append(nn.Linear(in_dim, out_nn[-1]))
        self.final_seg = nn.Sequential(*self.final_seg)

    def forward(self, pos, x, batch):
        neighbors, neighbors_batch = knn_graph(pos, self.k, batch)
        neighbors_xy, neighbors_batch_xy = knn_graph(pos[:,:2], self.k, batch)
        # neighbors, neighbors_batch = radius_graph(pos, 0.1, batch)

        # extract neighborhood features
        for pint in self.point_interact:
            x = pint(pos, x, batch, neighbors, neighbors_batch)  # b*n d

        # last point-wise processing to give final segmentation
        x = self.feature_cat(pos, x)
        x = self.final_seg(x)

        return x
