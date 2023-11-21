from tqdm.auto import tqdm
from itertools import product
from pathlib import Path
from multiprocessing import Pool
from functools import partial

# THIS HAS TO BE IMPORTED HERE OTHERWISE RASTERIO WILL CRASH IN POLYGON FILE torch-3dpoints-powerline/torch_points3d/core/data_transform/polygonCNN.py
import open3d as o3d
import numpy as np
import laspy
import torch
from torch_geometric.data import Data, Dataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.core.data_transform.polygonCNN import PolygonCNN
from torch_points3d.core.data_transform.outlier_removal_o3d import OutlierDetection
import ipdb

#CLASSES = ["Unclassified", "Ground", "Low veg", "Medium Veg", "High Veg", "Building", "noise", "keypoint", "Water", "wire_conductor", "bridge_deck", "Reserved" ]
CLASSES = ["others", "wire_conductor"]

# CLASSES = ["Ground", "High Veg", "Building", "combined"]
from torch_points3d.datasets.base_dataset import BaseDataset

class Denmark(Dataset):
    def __init__(self, root, processed_folder,
                 cnn_param, outlier_param, split, block_size, overlap: float,
                 global_z=None, transform=None, pre_transform=None, pre_filter=None):
        
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

        self.cnn_param = cnn_param
        self.outlier_param = outlier_param


        self.processed_split_folder = (Path(root) / str(processed_folder) / f"{split}_{overlap}_{block_size}")
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

        self.processed_file_names_ = list(self.processed_split_folder.glob("*cloud_*.pt"))
        #ipdb.set_trace()

        print("Total of {} samples in {} set.".format(len(self), split))

    def process(self):
        # ## process data
        print(f"processing {self.split} split")
        self.processed_split_folder.mkdir(exist_ok=True, parents=True)
        room_points, room_labels = [], []
        room_coord_min, room_coord_max = [], []
        room_names = []
        n_point_rooms = []

        file_names = [f.stem for f in self.raw_file_names]
        # load all room data
        new_laz_dir = Path(self.raw_dir).joinpath(self.split).joinpath("CNNLaz")
        new_laz_dir.mkdir(exist_ok=True)
        path_to_data = Path(self.raw_dir) / self.split

        outlier_clf = OutlierDetection(voxel_size=self.outlier_param["voxel_size"],
                                        nb_neighbors=self.outlier_param["nb_neighbors"], std_ratio=self.outlier_param["std_ratio"])
        CNNPreprocess = PolygonCNN(path_to_data=str(path_to_data), path_to_model=self.cnn_param["path_to_model"],
                                   network_size=self.cnn_param["network_size"], image_size=self.cnn_param["image_size"],
                                   meters_around_line=self.cnn_param["meters_around_line"], simplify_tolerance=self.cnn_param["simplify_tolerance"],
                                   cc_area=self.cnn_param["cc_area"])

        for file in file_names:
        # for file in tqdm(file_names):
            # ipdb.set_trace()
            new_laz = CNNPreprocess(file)
            new_laz = outlier_clf.RemoveOutliersFromLas(new_laz)
            new_laz.write(str(new_laz_dir)+'/'+file+".laz", do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)

        # load all room data
        new_laz_files = new_laz_dir.glob("*.laz")

        for room_path in tqdm(new_laz_files):
            #print(room_path)
            room_name = room_path.stem

            # TODO modify the laspy for reading the version
            try:
                room_data = laspy.read(room_path, laz_backend=laspy.compression.LazBackend.LazrsParallel)
            except Exception as e:
                continue

            room_data = np.stack([room_data.X, room_data.Y, room_data.Z, room_data.classification], 1).astype(np.float64)

            # split into points and labels
            points, tmp_labels = room_data[:, :-1], room_data[:, -1]  # xyzc, N*4

            # labels = tmp_labels
            labels = np.zeros_like(tmp_labels)

            if 14 not in tmp_labels:
                continue

            unclassified = tmp_labels == 1
            ground = tmp_labels == 2
            low_veg = tmp_labels == 3
            mid_veg = tmp_labels == 4
            hig_veg = tmp_labels == 5
            building = tmp_labels == 6
            noise = tmp_labels == 7

            keypoint = tmp_labels == 8
            water = tmp_labels == 9
            wire_conductor = tmp_labels == 14
            bridge_deck =  tmp_labels == 17
            Reserved =  tmp_labels == 18

            ## re name for labels
            labels[unclassified] = 0
            labels[ground] = 0
            labels[low_veg] = 0
            labels[mid_veg] = 0
            labels[hig_veg] = 0
            labels[building] = 0
            labels[noise] = 0
            labels[water] = 0
            labels[keypoint] = 0
            labels[wire_conductor] = 1
            labels[bridge_deck] = 0
            labels[Reserved] = 0

            # stats for normalization
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

            # save samples and stats
            room_points.append(points)
            room_labels.append(labels)
            room_coord_min.append(coord_min)
            room_coord_max.append(coord_max)
            room_names.append(room_name)
            n_point_rooms.append(labels.size)

        # assert len(n_point_rooms) > 0, f"No data at {str(Path(self.raw_dir) / self.split)}"

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
        # ipdb.set_trace()

        # partition rooms
        # each room is scaled between 0 and 1 so just try to have similar point counts
        counter = 0
        for room_i in tqdm(range(len(n_point_rooms))):
            # recover max room length from room and block scaling
            room_max = ((room_coord_max[room_i] - room_coord_min[room_i]) / room_coord_scale[room_i])[:2]
            n_split_2d = (np.ceil(room_max / self.overlap_difference)).astype(int)
            #ipdb.set_trace()
            for i, j in tqdm(product(range(n_split_2d[0]), range(n_split_2d[1]))):
                point_idx = self.query_room(room_points[room_i], i, j)

                if len(point_idx) > 0:
                    points = room_points[room_i][point_idx]
                    labels = room_labels[room_i][point_idx]
                    room_name = room_names[room_i]
    
                    torch.save({"filename": room_name, "coord_min": coord_min, "coord_max":coord_max,
                                "points": points, "labels": labels, "room_idx": room_i, "part_i": i, "part_j": j},
                                self.processed_split_folder / f"{room_name}_cloud_{counter}.pt")
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
        torch.save(self.processed_file_names_, self.processed_split_folder / "processed_file_names.pt")

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
        # features = torch.tensor(features, dtype=torch.float32)

        # data = Data(x=features, y=labels, pos=points).detach()
        data = Data( y=labels, pos=points)

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
        # ipdb.set_trace()
        return len(self.processed_file_names_)

    def __len__(self):
        return self.len()

    @property
    def raw_file_names(self):
        return list((Path(self.raw_dir) / self.split).glob("*.laz"))

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
        #ipdb.set_trace()
        cnn_param = {}
        cnn_param["path_to_model"] = dataset_opt.path_to_model
        cnn_param["network_size"] = dataset_opt.network_size
        cnn_param["image_size"] = dataset_opt.image_size
        cnn_param["meters_around_line"] = dataset_opt.meters_around_line
        cnn_param["simplify_tolerance"] = dataset_opt.simplify_tolerance
        cnn_param["cc_area"] = dataset_opt.cc_area

        outlier_param = {}
        outlier_param["voxel_size"] = dataset_opt.outlier_voxel_size
        outlier_param["nb_neighbors"] = dataset_opt.outlier_nb_neighbors
        outlier_param["std_ratio"] = dataset_opt.outlier_std_ratio

        self.train_dataset = Denmark(
            split='train', root=self._data_path, processed_folder=dataset_opt.processed_folder,
            cnn_param= cnn_param, outlier_param=outlier_param,
            overlap=dataset_opt.train_overlap, block_size=block_size,
            transform=self.train_transform, pre_transform=self.pre_transform
        )

        self.val_dataset = Denmark(
            split='val', root=self._data_path, processed_folder=dataset_opt.processed_folder,
            cnn_param= cnn_param, outlier_param=outlier_param, overlap=0,
            block_size=block_size, global_z=self.train_dataset.global_z,
            transform=self.val_transform, pre_transform=self.pre_transform
         )

        self.test_dataset = Denmark(
            split='test', root=self._data_path, processed_folder=dataset_opt.processed_folder,
            cnn_param= cnn_param, outlier_param=outlier_param, overlap=0,
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
