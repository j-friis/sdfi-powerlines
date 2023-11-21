import torch.nn.functional as F
import logging
from omegaconf import OmegaConf

from torch_points3d.models.regression.base import LOSSES
from torch_points3d.utils.config import ConvolutionFormatFactory
from torch_points3d.modules.PointNet import *
from torch_points3d.models.base_model import BaseModel
from torch_points3d.utils.model_building_utils.resolver_utils import flatten_dict

log = logging.getLogger(__name__)


class PointNet(BaseModel):
    def __init__(self, opt, model_type=None, dataset=None, modules=None):
        super().__init__(opt)

        self._opt = OmegaConf.to_container(opt)
        self._is_dense = ConvolutionFormatFactory.check_is_dense_format(self.conv_type)

        self._build_model()

        self.loss_names = ["loss_reg"]  # , "loss_internal"] TODO not really supported by frame work

        self.visual_names = ["data_visual"]

        self.loss_fn = LOSSES[opt.get("loss_fn", "smoothl1")]

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data
        if data.x is not None:
            self.input_features = torch.cat([data.pos, data.x], axis=-1)
        else:
            self.input_features = data.pos
        if data.y is not None:
            self.labels = data.y
        else:
            self.labels = None
        if not hasattr(data, "batch"):
            self.batch_idx = torch.zeros(self.labels.shape[0]).long()
        else:
            self.batch_idx = data.batch
        self.pointnet_reg.set_scatter_pooling(not self._is_dense)

    def _build_model(self):
        if not hasattr(self, "pointnet_reg"):
            self.pointnet_reg = PointNetReg(**flatten_dict(self._opt))

    def forward(self, *args, **kwargs):
        x = self.pointnet_reg(self.input_features, self.input.batch)
        self.output = x

        internal_loss = self.get_internal_loss()

        if self.labels is not None:
            self.loss_reg = self.loss_fn(self.output.flatten(), self.labels.flatten())
            self.loss_internal = (internal_loss if internal_loss.item() != 0 else 0) * 0.001
            self.loss = self.loss_reg + self.loss_internal

        self.data_visual = self.input
        self.data_visual.pred = self.output
        return self.output

    def backward(self):
        self.loss.backward()
