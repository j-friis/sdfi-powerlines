import logging

import torch
import torch.nn.functional as F

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.PointAtt import PointAttSeg, UPointAttSeg
from torch_points3d.utils.model_building_utils.resolver_utils import flatten_dict

log = logging.getLogger(__name__)


class PointAtt(BaseModel):
    def __init__(self, opt, type, dataset, modules_lib):
        super().__init__(opt)

        self.pointnet_seg = PointAttSeg(**flatten_dict(opt))

        self.loss_names = ["loss_seg"]

        self.visual_names = ["data_visual"]

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data
        if data.x is not None:
            self.input_features = torch.cat([data.pos, data.x], axis=-1) # b*n c
            self.input_features = self.input_features.view(len(data.grid_size), -1, self.input_features.shape[-1]) # b n c
            self.input_features = self.input_features.transpose(2, 1) # b c n

        if data.y is not None:
            self.labels = data.y
        if not hasattr(data, "batch"):
            self.batch_idx = torch.zeros(self.labels.shape[0]).long()
        else:
            self.batch_idx = data.batch

    def forward(self, *args, **kwargs):
        bs, _, n_pts = self.input_features.shape
        x = self.pointnet_seg(self.input_features)

        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.pointnet_seg.n_classes), dim=-1)
        x = x.view(bs * n_pts, self.pointnet_seg.n_classes)
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


# TODO this could be integrated in the above class
class UPointAtt(PointAtt):
    def __init__(self, opt, type, dataset, modules_lib):
        super().__init__(opt, type, dataset, modules_lib)

        self.pointnet_seg = UPointAttSeg(**flatten_dict(opt))