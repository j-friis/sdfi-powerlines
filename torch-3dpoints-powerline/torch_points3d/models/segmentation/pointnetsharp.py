import logging

import torch
import torch.nn.functional as F

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.PointNetSharp import PointNetSharpSeg
from torch_points3d.utils.model_building_utils.resolver_utils import flatten_dict

log = logging.getLogger(__name__)


class PointNetSharp(BaseModel):
    def __init__(self, opt, type, dataset, modules_lib):
        super().__init__(opt)

        self.point_seg = PointNetSharpSeg(**flatten_dict(opt))

        self.loss_names = ["loss_seg"]

        self.visual_names = ["data_visual"]

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data
        if data.y is not None:
            self.labels = data.y
        if not hasattr(data, "batch"):
            self.batch_idx = torch.zeros(self.labels.shape[0]).long()
        else:
            self.batch_idx = data.batch

    def forward(self, *args, **kwargs):
        x = self.point_seg(self.input.pos, self.input.x, self.input.batch)

        self.output = x

        if self.labels is not None:
            self.loss_seg = F.cross_entropy(
                self.output, self.labels, ignore_index=IGNORE_LABEL
            )
            self.loss = self.loss_seg

        self.data_visual = self.input
        self.data_visual.pred = torch.max(self.output, -1)[1]
        return self.output

    def backward(self):
        self.loss.backward()
