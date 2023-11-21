from tqdm.auto import tqdm
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

# CLASSES = ["Ground", "Low veg", "Medium Veg", "High Veg", "Building", "Water"]
CLASSES = ["Ground", "High Veg", "Building", "combined"]
from torch_points3d.datasets.base_dataset import BaseDataset


class Denmark(Dataset):
    def __init__(self, root, split, block_size, overlap: float, global_z=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.processed_file_names_ = []
        # init some constant to know label names
        self.classes = CLASSES
        if isinstance(block_size, float):
            block_size = [block_size] * 2
        self.block_size = np.array(block_size)
        self.global_z = global_z
        self.split = split

        # this works without taking dataset x, y scaling into account since we already scaled to (0, 1)
        self.overlap = overlap  # defines an overlap ratio
        self.overlap_value = np.array([overlap] * 2)
        self.overlap_difference = 1 - self.overlap_value

        self.n_classes = len(CLASSES)

        self.processed_split_folder = (Path(root) / "processed" / f"{split}_{overlap}_{block_size}")
        super(Denmark, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        print(f"loading processed {split} split")
        stats = torch.load(self.processed_split_folder / "stats.pt")
        self.room_names = stats["room_names"]
        self.room_coord_min = stats["room_coord_min"]
        self.room_coord_max = stats["room_coord_max"]
        self.room_coord_scale = stats["room_coord_scale"]
        self.global_z = stats["global_z"]

        self.processed_file_names_ = list(self.processed_split_folder.glob("cloud_*.pt"))

        print("Total of {} samples in {} set.".format(len(self), split))

    def process(self):
        # ## process data
        print(f"processing {self.split} split")
        self.processed_split_folder.mkdir(exist_ok=True)
        room_points, room_labels = [], []
        room_coord_min, room_coord_max = [], []
        room_names = []
        n_point_rooms = []
        # load all room data
        for room_path in tqdm(self.raw_file_names):
            print(room_path)
            room_name = room_path.stem

            # codes for three classes
            # load data (pandas is way faster than numpy in this regard)
            room_data = pd.read_csv(room_path, sep=" ", header=None)  # xyzc, N*4

            room_data = room_data.values

            # split into points and labels
            points, tmp_labels = room_data[:, :-1], room_data[:, -1]  # xyzc, N*4

            labels = np.zeros_like(tmp_labels)
            # class 0 is ground
            hig_veg = tmp_labels == 3
            building = tmp_labels == 4
            ground = tmp_labels == 0
            labels[hig_veg] = 1  # high veg
            labels[building] = 2  # building
            labels[~(hig_veg | building | ground)] = 3  # rest

            # stats for normalization
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

            # save samples and stats
            room_points.append(points)
            room_labels.append(labels)
            room_coord_min.append(coord_min)
            room_coord_max.append(coord_max)
            room_names.append(room_name)
            n_point_rooms.append(labels.size)
        assert len(n_point_rooms) > 0, f"No data at {str(Path(self.raw_dir) / self.split)}"
        # give global_z from training set
        global_z = self.global_z
        if global_z is None:
            # choose the room with biggest spatial gap in height for scaling the z-axis
            diff = np.array(room_coord_max)[:, 2] - np.array(room_coord_min)[:, 2]
            max_room = np.argmax(diff, 0)
            global_z = np.array(room_coord_min)[max_room, 2], np.array(room_coord_max)[max_room, 2]
        min_z, max_z = global_z
        block_scale = np.concatenate([self.block_size, [1.]])
        room_coord_scale = []
        for points, coord_min, coord_max in zip(room_points, room_coord_min, room_coord_max):
            # override local z with global z
            coord_min[2] = min_z
            coord_max[2] = max_z
            # apply min-max normalization
            # center (makes it easier to do the block queries)
            points[:] = points - coord_min
            # scale
            room_scale = (coord_max - coord_min)  # this shouldn't change for our 1k dataset
            points[:] = points / (room_scale * block_scale)
            room_coord_scale.append(room_scale * block_scale)
        print(f"saving processed {self.split} split")
        # partition rooms
        # each room is scaled between 0 and 1 so just try to have similar point counts
        counter = 0
        for room_i in tqdm(range(len(n_point_rooms))):
            # recover max room length from room and block scaling
            room_max = ((room_coord_max[room_i] - room_coord_min[room_i]) / room_coord_scale[room_i])[:2]
            n_split_2d = (np.ceil(room_max / self.overlap_difference)).astype(int)
            for i, j in tqdm(product(range(n_split_2d[0]), range(n_split_2d[1]))):
                point_idx = self.query_room(room_points[room_i], i, j)

                if len(point_idx) > 0:
                    points = room_points[room_i][point_idx]
                    labels = room_labels[room_i][point_idx]

                    torch.save({"points": points, "labels": labels, "room_idx": room_i, "part_i": i, "part_j": j},
                                self.processed_split_folder / f"cloud_{counter}.pt")
                    counter += 1
            # TODO test how many points are actually in the partitions and merge/expand them if necessary
        stats = {
            "room_names": room_names,
            "room_coord_min": room_coord_min,
            "room_coord_max": room_coord_max,
            "room_coord_scale": room_coord_scale,
            "global_z": global_z
        }
        torch.save(stats, self.processed_split_folder / "stats.pt")

    # def get_partition_index:
    def get(self, idx):
        file = self.processed_file_names_[idx]
        cloud = torch.load(file)
        points = cloud["points"]
        labels = cloud["labels"]

        # normalize per block
        points = center_block(points)

        points = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        data = Data(y=labels, pos=points).detach()

        return data

    def query_room(self, points, i, j):
        block_min = np.array([i * self.overlap_difference[0], j * self.overlap_difference[1]])
        block_max = np.array([
            (i + 1) * self.overlap_difference[0] + self.overlap_value[0],
            (j + 1) * self.overlap_difference[1] + self.overlap_value[1]
        ])
        point_idxs = np.where(
            (points[:, 0] >= block_min[0]) & (points[:, 0] < block_max[0])
            & (points[:, 1] >= block_min[1]) & (points[:, 1] < block_max[1])
        )[0]
        return point_idxs

    def len(self):
        return len(self.processed_file_names_)

    def __len__(self):
        return self.len()

    @property
    def raw_file_names(self):
        return list((Path(self.raw_dir) / self.split).glob("*.txt"))

    @property
    def processed_file_names(self):
        return self.processed_file_names_ + [self.processed_split_folder / "stats.pt"]

    def download(self):
        pass

    @property
    def num_classes(self) -> int:
        return self.n_classes

    def get_global_z(self):
        return self.room_coord_min[0][2], self.room_coord_max[0][2]

    def get_descaled_dataset(self):
        ''' descale data with saved stats '''
        rooms = []
        for points, c_min, c_scale in zip(self.room_points, self.room_coord_min, self.room_coord_scale):
            points = np.copy(points)
            points[:, :3] = points[:, :3] * c_scale + c_min

            rooms.append(points)
        return rooms


def center_block(points):
    # center the samples around the center point
    selected_points = np.copy(points)  # make sure we work on a copy
    selected_points -= np.median(selected_points, 0)

    return selected_points


class DenmarkDataset(BaseDataset):
    """
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        block_size = (dataset_opt.block_size_x,
                      dataset_opt.block_size_y)  # tuple for normalized sampling area (e.g., if 1km = 1, 200m = 0.2)
        self.train_dataset = Denmark(
            split='train', root=self._data_path, overlap=dataset_opt.train_overlap,
            block_size=block_size,
            transform=self.train_transform, pre_transform=self.pre_transform
        )

        self.val_dataset = Denmark(
            split='val', root=self._data_path, overlap=0,
            block_size=block_size, global_z=self.train_dataset.global_z,
            transform=self.val_transform, pre_transform=self.pre_transform
        )

        self.test_dataset = Denmark(
            split='test', root=self._data_path, overlap=0,
            block_size=block_size, global_z=self.train_dataset.global_z,
            transform=self.test_transform, pre_transform=self.pre_transform
        )

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @property  # type: ignore
    def stuff_classes(self):
        """ Returns a list of classes that are not instances
        """
        return self.train_dataset.stuff_classes

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """

        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
