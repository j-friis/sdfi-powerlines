import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import argparse
import cv2
import glob
import laspy
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, mapping
import shapely
from matplotlib.path import Path
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
import os
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d

class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, path="~/data/", voxel_size=0.5, nb_neighbors=10, std_ratio=20):
        self.path = path
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.voxel_size = voxel_size
        self.predictions = []
        self.loaded_files = []
        self.all_paths = []


    def GenPath(self):
        if self.path[-1] == '/':
            return self.path
        else:
            return self.path+'/'

    def GetPathRelations(self):
        full_path_to_data = self.GenPath()
        for path in glob.glob(full_path_to_data+'new_las/*'):
            self.all_paths.append(path)
        self.all_paths.sort()
        
    def fit(self, X, y):
        print(self.get_params())
        self.GetPathRelations()
        for p in self.all_paths:
            tmp_las = laspy.read(p, laz_backend=laspy.compression.LazBackend.LazrsParallel)
            point_data = np.stack([tmp_las.X, tmp_las.Y, tmp_las.Z], axis=0).transpose((1, 0))

            # Create o3d Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_data)
            voxel_down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

            _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                                std_ratio=self.std_ratio)
            self.loaded_files.append(tmp_las)
            self.predictions.append(ind)
    
    def score(self, _, __):

        all_pct_data_kept = []
        all_pct_wires_removed = []
        
        for i, tmp_las in enumerate(self.loaded_files):
            inlier_mask = self.predictions[i]
            masked_las = tmp_las[inlier_mask]

            wires_in_las = np.sum(tmp_las.classification == 14)
            wires_after_removal = np.sum(masked_las.classification == 14)

            data_size = len(tmp_las)
            data_size_after_removal = len(masked_las)
            
            pct_wires_removed = 0
            if wires_in_las > 0:
                pct_wires_removed = 1-wires_after_removal/wires_in_las
            
            pct_data_kept = data_size_after_removal/data_size
            all_pct_data_kept.append(pct_data_kept)
            all_pct_wires_removed.append(pct_wires_removed)

            file = open('results_outliers_o3d.txt', 'a')
            items = [i, self.all_paths[i], 1-pct_wires_removed, 1-pct_data_kept, data_size, wires_in_las, data_size_after_removal, wires_in_las-wires_after_removal,self.get_params()]
            
            for item in items[:-1]:
                file.write(str(item)+",")
            file.write(str(items[-1])+"\n")
            file.close()

        epsilon = 0.001
        score = 0
        if np.mean(all_pct_wires_removed) <= epsilon:
            score = 1-np.mean(all_pct_data_kept)
        
        print("Finished Iter with score: ", score)
        return score

parser = argparse.ArgumentParser(description='Path to data folder.')
parser.add_argument('folder', type=str, help='folder with data')
args = parser.parse_args()
dir = args.folder

if __name__ == "__main__":

    n_cpu = os.cpu_count()
    print("Number of CPUs in the system:", n_cpu)

    cv = [(slice(None), slice(None))]
    params = {
            "path": Categorical([dir]),
            "voxel_size": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "nb_neighbors": Integer(5, 20),
            "std_ratio": Integer(15, 50),
    }

    opt = BayesSearchCV(
        TemplateClassifier(),search_spaces=params,
        cv=cv,
        n_iter=50,
        n_jobs=n_cpu-1,
        random_state=0
    )
    
    # executes bayesian optimization
    X = [[1,2],[1,2],[1,2],[1,2]]
    Y = [1,2,1,2]

    print("Started Fit")
    _ = opt.fit(X,Y)

    # model can be saved, used for predictions or scoring
    print("The best score: ", opt.best_score_)
    print("The best params ", opt.best_params_)
