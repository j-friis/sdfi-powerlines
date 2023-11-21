import numpy as np
import open3d as o3d
#from open3d.geometry import PointCloud

class OutlierDetection():

    def __init__(self, voxel_size=0.7, nb_neighbors=19, std_ratio=50):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.voxel_size = voxel_size
        self.removed_outliers = 0
        self.processed_point_clouds = 0

    def RemoveOutliersFromPointData(self, point_data):
        # Create o3d Point Cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_data)
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                                std_ratio=self.std_ratio)
        self.removed_outliers += len(point_data)-len(ind)
        self.processed_point_clouds += 1
        return ind
    
    
    def RemoveOutliersFromLas(self, las):
        point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
        ind = self.RemoveOutliersFromPointData(point_data)
        return las[ind]
    



