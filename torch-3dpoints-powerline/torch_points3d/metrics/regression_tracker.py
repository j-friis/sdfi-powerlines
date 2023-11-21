from typing import Dict, Any
import torch
import torch.nn.functional as F
from torchnet.meter import MSEMeter

from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters.maemeter import MAEMeter
from torch_points3d.metrics.meters.r2meter import R2Meter
from torch_points3d.models import model_interface


class RegressionTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        """ This is a generic tracker for segmentation tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch
        Arguments:
            dataset  -- dataset to track (used for the number of classes)
        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(RegressionTracker, self).__init__(stage, wandb_log, use_tensorboard)

        self.n_targets = dataset.num_classes
        self.name_targets = dataset.targets
        self.target_means = dataset.get_mean_targets()

        # for r2 score
        self.reset(stage)
        self._metric_func = {
            "rmse": min,
            "mae": min,
            "r2": max,
            "loss_regr": min,
            "loss": min,
        }  # Those map subsentences to their optimization functions

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._rmse = [MSEMeter(root=True) for _ in range(self.n_targets + 1)]
        self._mae = [MAEMeter() for _ in range(self.n_targets + 1)]
        self._r2 = [R2Meter(self.target_means[stage][i]) for i in range(self.n_targets)]

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = model.get_output()
        targets = model.get_labels().reshape(-1, self.n_targets)

        for i in range(self.n_targets):
            out = outputs[:, i]
            target = targets[:, i]
            self._rmse[i].add(out, target)
            self._mae[i].add(out, target)
            self._r2[i].add(out, target)
        self._rmse[-1].add(outputs, targets)
        self._mae[-1].add(outputs, targets)

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        for i in range(self.n_targets):
            metrics[f"{self._stage}_{self.name_targets[i]}_rmse"] = self._rmse[i].value()
            metrics[f"{self._stage}_{self.name_targets[i]}_mae"] = self._mae[i].value()
            metrics[f"{self._stage}_{self.name_targets[i]}_r2"] = self._r2[i].value()

        metrics[f"{self._stage}_total_rmse"] = self._rmse[-1].value()
        metrics[f"{self._stage}_total_mae"] = self._mae[-1].value()
        return metrics

    @property
    def metric_func(self):
        return self._metric_func
