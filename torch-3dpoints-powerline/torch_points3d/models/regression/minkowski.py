import logging
import torch
import torch.nn as nn

from torch_points3d.models.regression.base import LOSSES
from torch_points3d.modules.MinkowskiEngine import initialize_minkowski_unet
from torch_points3d.models.base_model import BaseModel
import MinkowskiEngine as ME

log = logging.getLogger(__name__)


class Minkowski_Baseline_Model(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model, self).__init__(option)
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.num_classes, activation=option.activation,
            first_stride=option.first_stride, dropout=option.dropout, global_pool=option.global_pool,
            statpooling=option.statpooling, # bias=option.bias,
            **option.get("extra_options", {})
        )
        self.loss_names = ["loss_regr"]
        self.visual_names = ["data_visual"]
        self.loss_fn = LOSSES[option.get("loss_fn", "smoothl1")]

    def set_input(self, data, device):
        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.data_visual = data
        self.input = ME.SparseTensor(features=data.x, coordinates=coords, device=device)
        self.labels = data.y.to(device)

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input).features
        if self.labels is not None:
            self.loss = self.loss_regr = self.loss_fn(self.output.flatten(), self.labels.flatten())

        self.data_visual.pred = self.output

    def backward(self):
        self.loss.backward()


class Minkowski_Exponential_Model(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Exponential_Model, self).__init__(option)
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.num_classes, activation=option.activation,
            first_stride=option.first_stride, dropout=option.dropout, global_pool=option.global_pool,
            **option.get("extra_options", {})
        )
        self.loss_names = ["loss_regr"]
        self.visual_names = ["data_visual"]
        self.loss_fn = LOSSES[option.get("loss_fn", "l2")]

        self.exp_model = nn.Linear(len(dataset.stats), dataset.num_classes)

        torch.nn.init.xavier_normal_(self.exp_model.weight, 0.1)
        torch.nn.init.zeros_(self.exp_model.bias)

    def set_input(self, data, device):
        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.bs = len(data.grid_size)
        self.data_visual = data
        self.input = ME.SparseTensor(features=data.x, coordinates=coords, device=device)
        self.stats = data.stats.to(device)
        self.labels = data.y.to(device)

    def forward(self, *args, **kwargs):
        model_out = self.model(self.input).features
        stats = self.stats.reshape(self.bs, -1)
        self.exp_output = torch.expm1(self.exp_model(torch.log1p(stats)))
        self.output = self.exp_output.detach() + model_out

        if self.labels is not None:
            self.loss_exp_regr = self.loss_fn(self.exp_output.flatten(), self.labels.flatten())
            self.loss = self.loss_regr = self.loss_fn(self.output.flatten(), self.labels.flatten()) + self.loss_exp_regr
        self.data_visual.pred = self.output

    def backward(self):
        self.loss.backward()
