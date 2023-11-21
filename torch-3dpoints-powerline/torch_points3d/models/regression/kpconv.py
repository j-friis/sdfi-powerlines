from typing import Any
import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Linear
import torch.nn.functional as F
from torch import nn

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.models.regression.base import LOSSES
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.models.base_architectures.backbone import BackboneBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

log = logging.getLogger(__name__)


def global_maxmean_pool(x, batch):
    dim = x.size(1)
    gmax = global_max_pool(x[:, :dim // 2], batch)
    gmean = global_mean_pool(x[:, dim // 2:], batch)
    return torch.cat([gmax, gmean], dim=1)


GLOBAL_POOL = {
    "max": global_max_pool,
    "add": global_add_pool,
    "mean": global_mean_pool,
    "maxmean": global_maxmean_pool,
}


class KPConv(BackboneBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._num_outputs = dataset.num_classes

        # Assemble encoder / decoder
        BackboneBasedModel.__init__(self, option, model_type, dataset, modules)
        # Build final MLP
        last_mlp_opt = option.mlp_cls
        in_feat = last_mlp_opt.nn[0]
        self.FC_layer = Sequential()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                        FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                        LeakyReLU(0.2),
                    ]
                ),
            )
            in_feat = last_mlp_opt.nn[i]

        if last_mlp_opt.dropout:
            self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.add_module("Target", Lin(in_feat, self._num_outputs, bias=False))
        self.loss_names = ["loss_regr"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])

        self.visual_names = ["data_visual"]

        self.loss_fn = LOSSES[option.get("loss_fn", "smoothl1")]

        self.global_pool_fn = GLOBAL_POOL[option["global_pool"]]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        data = data.to(device)
        data.x = add_ones(data.pos, data.x, True)

        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
            del data.upsample
            del data.multiscale
        else:
            self.upsample = None
            self.pre_computed = None

        self.input = data
        self.labels = data.y
        self.batch_idx = data.batch

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        data = self.input
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, precomputed=self.pre_computed)
            stack_down.append(data)

        data = self.down_modules[-1](data, precomputed=self.pre_computed)

        last_feature = self.global_pool_fn(data.x, data.batch)

        self.output = self.FC_layer(last_feature)

        if self.labels is not None:
            self.compute_loss()

        self.data_visual = self.input
        self.data_visual.pred = self.output
        return self.output

    def compute_loss(self):

        self.loss = 0

        # Get regularization on weights
        if self.lambda_reg:
            self.loss_reg = self.get_regularization_loss(regularizer_type="l2", lambda_reg=self.lambda_reg)
            self.loss += self.loss_reg

        # Collect internal losses and set them with self and them to self for later tracking
        if self.lambda_internal_losses:
            self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        # Final cross entrop loss
        self.loss_regr = self.loss_fn(
            self.output.flatten(),
            self.labels.flatten()
        )
        self.loss += self.loss_regr

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G
