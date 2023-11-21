import laspy
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Dataset, Data

from tqdm.auto import tqdm

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.regression_tracker import RegressionTracker


class Biomass(Dataset):
    def __init__(
            self, root, file_names, split, stats=None, store_samples=False,
            x_radius=15., y_radius=15., z_radius=20., same_year_only=False,
            transform=None, targets=None, features=None, feature_scaling=None,
            first_return_only=False, pre_transform=None, pre_filter=None,
    ):
        if isinstance(file_names, str):
            file_names = [file_names]
        self.root = root
        self.split = split
        self.store_samples = store_samples

        (Path(self.processed_dir) / self.split / "done.flag")

        # no need to load dataframe if processed data is present
        if store_samples and (Path(self.processed_dir) / self.split / "done.flag").exists():
            self.num_samples = len(list((Path(self.processed_dir) / self.split).glob("*.pt")))
            self.raw_file_names_ = []
        else:
            self.df = []
            for file_name in file_names:
                self.df.append(pd.read_csv(Path(root) / "raw" / file_name))
            self.df = pd.concat(self.df, axis=0)

            self.df.eval("temp_diff_years = temp_diff_days / 365", inplace=True)
            self.df.eval("exp_temp_diff_years = expm1(temp_diff_years)", inplace=True)
            self.same_year_only = same_year_only
            if same_year_only:
                self.df.query("not is_1y_distant", inplace=True)
            self.num_samples = len(self.df)
            self.raw_file_names_ = self.df.las_file.values

        self.targets = targets
        self.features = [] if features is None else features
        self.stats = [] if stats is None else stats
        # difference between measurement and pointclouds taken
        self.radius = np.array([[x_radius, y_radius, z_radius]])

        # if not give, calculate on given data
        if feature_scaling is None:
            feature_scaling = {  # feature: (center, scale)
                # "intensity": (
                #     np.median(self.df["intensity_q50"]),
                #     np.median(self.df["intensity_q50"] - self.df["intensity_q75"]) * 1.349
                # ),
                # "classification": (
                #     np.median(self.df["classification_min"]),
                #     np.median(self.df["classification_max"] - self.df["classification_min"])
                # ),
                # "red": (
                #     np.median(self.df["red_q50"]),
                #     np.median(self.df["red_q50"] - self.df["red_q75"]) * 1.349
                # ),
                # "green": (
                #     np.median(self.df["green_q50"]),
                #     np.median(self.df["green_q50"] - self.df["green_q75"]) * 1.349
                # ),
                # "blue": (
                #     np.median(self.df["blue_q50"]),
                #     np.median(self.df["blue_q50"] - self.df["blue_q75"]) * 1.349
                # ),
                "num_returns": (0., 5.),
                "return_num": (0., 5.),
            }
        self.feature_scaling = feature_scaling
        self.first_return_only = first_return_only

        super(Biomass, self).__init__(
            root, transform, pre_transform, pre_filter
        )

    @property
    def raw_file_names(self):
        path = Path(self.raw_dir)
        return [path / f for f in self.raw_file_names_]

    # TODO they are the same for now
    @property
    def processed_file_names(self):
        if self.store_samples:
            path = Path(self.processed_dir)
            files = [(path / self.split / f"{i}.pt") for i in range(self.num_samples)]
        else:
            path = Path(self.raw_dir)
            files = [path / f for f in self.df.las_file.values]
        return files

    def download(self):
        pass
        # TODO could load directly
        # # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)

    def process(self):
        flag = (Path(self.processed_dir) / self.split / "done.flag")

        if self.store_samples and not flag.exists():
            (Path(self.processed_dir) / self.split).mkdir(exist_ok=True)
            for idx, file in tqdm(enumerate(self.processed_file_names)):
                if file.exists():
                    continue
                data = self.get_(idx)
                torch.save(data, file)
            flag.touch()

    @property
    def num_classes(self) -> int:
        return len(self.targets)

    def len(self):
        return self.num_samples

    def get(self, idx):
        if self.store_samples:
            data = torch.load(self.processed_file_names[idx])
        else:
            data = self.get_(idx)
        return data

    def get_(self, idx):
        df = self.df.iloc[idx]

        las = laspy.read(Path(self.raw_dir) / df.las_file)
        # only coordinates for now
        x = np.stack([las.x, las.y, las.z], 1)
        too_high = x[:, 2] > 50  # if higher, they are too high

        features = []
        for feature in self.features:
            if feature == "temp_diff_years":
                feat = np.array([df.temp_diff_years] * len(x))
            else:
                feat = getattr(las, feature)
            center, scale = self.feature_scaling.get(feature, (0., 1.))
            features.append((feat - center) / scale)

        # interesting features for each point
        if len(features) > 0:
            features = np.stack(features, 1).astype(np.float)
        else:
            features = np.zeros((len(x), 0))

        # normalize
        x_center = (np.quantile(x, 0.99, axis=0, keepdims=True) + np.quantile(x, 0.01, axis=0, keepdims=True)) / 2
        # do not center height (0 should always be ground, which is vital information)
        x_center[0, 2] = 0
        x = (x - x_center) / self.radius

        # target
        y = df[self.targets]

        # global stats
        stats = df[self.stats]

        # remove too high values
        x = x[~too_high]
        features = features[~too_high]

        if self.first_return_only:
            only_first_return = np.array(las.return_number)[~too_high] == 1
            x = x[only_first_return]
            features = features[only_first_return]

        # to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        stats = torch.tensor(stats, dtype=torch.float32)

        data = Data(x=features, y=y, pos=x, stats=stats)

        return data


class BiomassDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.merge_val_set = dataset_opt.merge_val_set
        self.targets = dataset_opt.targets
        self.features = dataset_opt.features
        self.stats = dataset_opt.stats

        train_files = "train_split.csv"
        if self.merge_val_set:
            train_files = [train_files, "val_split.csv"]

        self.train_dataset = Biomass(
            self._data_path, file_names=train_files, split="train", same_year_only=dataset_opt.same_year_only,
            targets=self.targets, features=self.features, first_return_only=dataset_opt.first_return_only,
            stats=dataset_opt.stats, transform=self.train_transform, pre_transform=self.pre_transform,
            store_samples=dataset_opt.store_samples
        )
        if not self.merge_val_set:
            self.val_dataset = Biomass(
                self._data_path, file_names="val_split.csv", split="val",
                targets=self.targets, features=self.features, feature_scaling=self.train_dataset.feature_scaling,
                first_return_only=dataset_opt.first_return_only, store_samples=dataset_opt.store_samples,
                stats=dataset_opt.stats, transform=self.val_transform, pre_transform=self.pre_transform
            )
        self.test_dataset = Biomass(
            self._data_path, file_names="test_split.csv", split="test",
            targets=self.targets, features=self.features, feature_scaling=self.train_dataset.feature_scaling,
            first_return_only=dataset_opt.first_return_only, store_samples=dataset_opt.store_samples,
            stats=dataset_opt.stats, transform=self.test_transform, pre_transform=self.pre_transform
        )

        if dataset_opt.store_samples:
            means_file = (Path(self.train_dataset.processed_dir) / "mean_targets.pt")
            if means_file.exists():
                self.mean_targets_ = torch.load(means_file)
            else:
                self.mean_targets_ = self.get_mean_targets_()
                torch.save(self.mean_targets_, means_file)
        else:
            self.mean_targets_ = self.get_mean_targets_()

    def get_mean_targets(self):
        return self.mean_targets_

    def get_mean_targets_(self):
        dict = {
            "train": np.mean(self.train_dataset.df[self.targets].values, 0),
            "test": np.mean(np.concatenate([tdata.df[self.targets].values for tdata in self.test_dataset], 0), 0),
        }
        if not self.merge_val_set:
            dict.update({"val": np.mean(self.val_dataset.df[self.targets].values, 0)})

        return dict

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return RegressionTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

# TODO adjust this
# def test():
#     import sys
#     try:
#         root = Path(sys.argv[1])
#     except IndexError:
#         raise Exception("please give the path to the files as first argument")
#
#     root = Path(sys.argv[1])
#     dataset = Biomass(root, "train_split.csv")
#     for i, (x, y, features, machine, year_diff, las_file) in enumerate(dataset):
#         print(f"sample: {i}\n"
#               f"\tmax values \tx: {x.max(0)}\n"
#               f"\tmin values \tx: {x.min(0)}\n"
#               f"\tbio mass \t{y}\n"
#               f"\tmachine \t {machine}\n"
#               f"\tyear diff \t {year_diff}\n"
#               f"\tfile \t {las_file}")
#
#         if i == 10:
#             break
# if __name__ == "__main__":
#     test()
