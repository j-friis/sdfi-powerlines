import argparse
import json
from pathlib import Path
import glob
import os

## import the tools
from pathlib import Path
import os
import numpy as np
import pandas as pd
import laspy

import torch
## to find the neighbor points prediction
from sklearn.neighbors import KDTree
import numpy as np


## import the model tools
from torch_geometric.transforms import Compose
from torch_points3d.core.data_transform import MinPoints, XYZFeature, AddFeatsByKeys, GridSampling3D
from torch_points3d.core.data_transform.features import AddOnes
from torch_points3d.applications.pretrained_api import PretainedRegistry
from torch_geometric.data import Batch
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix

def get_nearest_neighbors(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""
    tree = KDTree(candidates, leaf_size=20, metric='euclidean')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    closest = np.squeeze(indices)
    return closest


def load_model(model_path: str, data_path: str):
    model = torch.load(model_path)
    model['run_config']['data']['dataroot'] = data_path
    torch.save(model, model_path)
    print(model['run_config']["data"])
    print(model['run_config']["data"]["train_transform"])

    ## transformer for non ones
    # pos_z = [ "pos_z" ]
    # list_add_to_x = [ True ]
    # delete_feats = [ True ]
    # first_subsampling = model['run_config']["data"]["first_subsampling"]
    # transform_test = Compose([MinPoints(512),
    #                     XYZFeature(add_x=False, add_y=False, add_z= True),
    #                     AddFeatsByKeys(list_add_to_x=list_add_to_x, feat_names= pos_z,delete_feats=delete_feats),
    #                     GridSampling3D(mode='last', size=first_subsampling, quantize_coords=True)
    #                     ])

    ## transformer for ones
    pos_z = [ "ones" ]
    list_add_to_x = [ True ]
    delete_feats = [ True ]
    first_subsampling = model['run_config']["data"]["first_subsampling"]
    input_nc_feats = [1]

    transform_test = Compose([MinPoints(512),
                     AddOnes(),
                     AddFeatsByKeys(list_add_to_x=list_add_to_x, feat_names= pos_z,delete_feats=delete_feats, input_nc_feats=input_nc_feats),
                     GridSampling3D(mode='last', size=first_subsampling, quantize_coords=True)
                     ])
    ### ['latest', 'loss_seg', 'acc', 'macc', 'miou']
    model_pl = PretainedRegistry.from_file(model_path, weight_name="miou").cuda()
    return model_pl, transform_test, model['run_config']['data']


def predict(room_info, model, filename, transform_test, predict_folder):
    ## loop for every files
    room_coord_mins = room_info['room_coord_min']
    room_coord_scales = room_info['room_coord_scale']
    files = list(glob.glob(predict_folder + f"/*{filename}*cloud*pt"))

    pred_data = []

    for file in files:
        sample = os.path.join(predict_folder, file)
        pt_data = torch.load(sample)
        room_index = pt_data['room_idx']

        room_coord_scale = room_coord_scales[room_index]
        pos_ = pt_data['points']
        point_in_original_las = pos_ * room_coord_scale + room_coord_mins[room_index]

        data_s = transform_test(Batch(pos=torch.from_numpy(pos_).float()))
        data_s.batch = torch.zeros(len(data_s.pos))
        data_s.y = torch.zeros(data_s.pos.shape[0]).long()
        index_to_nearst_neighbor = get_nearest_neighbors(pos_, data_s.pos)


        with torch.no_grad():
            model.eval()
            model.set_input(data_s, "cuda")
            model.forward(data_s)
        
        pre = model.output.cpu().numpy()
        m = torch.nn.functional.softmax(torch.tensor(pre), dim=1)
        cla_pre = np.argmax(m, axis=1)
        pre_ori = np.arange(len(pos_))
        if len(pos_) == 1:
            pre_ori[0] = cla_pre[0]
        else:
            for i in pre_ori:
                pre_ori[i] = cla_pre[index_to_nearst_neighbor[i]]
        combine_pre = np.column_stack((point_in_original_las, pre_ori.T))

        pred_data.append(combine_pre)

    pred_data = np.array([item for sublist in pred_data for item in sublist])

    return pred_data

def get_metrics(pred_data, model_las, whole_las):
    ## read original las file

    powerline_pts = pred_data[np.where(pred_data[:,3] == 1)].copy()
    powerline_pts_coord = powerline_pts[:,:-1].astype(np.int32) 

    model_las_point_data = np.stack([model_las.X, model_las.Y, model_las.Z], axis=0).transpose((1, 0))
    model_las_idx = get_nearest_neighbors(powerline_pts_coord, model_las_point_data)
    pred = np.zeros(len(model_las_point_data))
    model_las_label = model_las.classification 
    model_las_label = model_las_label==14 
    pred[model_las_idx] = 1
    pred = np.asarray(pred,dtype=np.int16)
    model_las_label = np.asarray(model_las_label,dtype=np.int16)

    cfm = ConfusionMatrix()
    #Use as count_predicted_batch(true pred)
    cfm.count_predicted_batch(model_las_label, pred)
    model_las_metric_dict = {}

    model_las_metric_dict["acc"] = 100 * cfm.get_overall_accuracy()
    model_las_metric_dict["macc"] = 100 * cfm.get_mean_class_accuracy()
    model_las_metric_dict["miou"] = 100 * cfm.get_overall_accuracy()
    model_las_metric_dict["miou_class"] = {
        i: "{:.2f}".format(100 * v)
        for i, v in enumerate(cfm.get_intersection_union_per_class()[0])
    }
    model_las_metric_dict["precision"] = cfm.get_confusion_matrix()[1][1]/(cfm.get_confusion_matrix()[1][1]+cfm.get_confusion_matrix()[0][1])
    model_las_metric_dict["recall"] = cfm.get_confusion_matrix()[1][1]/(cfm.get_confusion_matrix()[1][1]+cfm.get_confusion_matrix()[1][0])


    whole_las_point_data = np.stack([whole_las.X, whole_las.Y, whole_las.Z], axis=0).transpose((1, 0))
    whole_las_idx = get_nearest_neighbors(powerline_pts_coord, whole_las_point_data)
    pred = np.zeros(len(whole_las_point_data))
    whole_las_label = whole_las.classification 
    whole_las_label = whole_las_label==14 
    pred[whole_las_idx] = 1
    pred = np.asarray(pred,dtype=np.int16)
    whole_las_label = np.asarray(whole_las_label,dtype=np.int16)

    cfm = ConfusionMatrix()
    #Use as count_predicted_batch(true pred)
    cfm.count_predicted_batch(whole_las_label, pred)
    whole_las_metric_dict = {}

    whole_las_metric_dict["acc"] = 100 * cfm.get_overall_accuracy()
    whole_las_metric_dict["macc"] = 100 * cfm.get_mean_class_accuracy()
    whole_las_metric_dict["miou"] = 100 * cfm.get_overall_accuracy()
    whole_las_metric_dict["miou_class"] = {
        i: "{:.2f}".format(100 * v)
        for i, v in enumerate(cfm.get_intersection_union_per_class()[0])
    }
    whole_las_metric_dict["precision"] = cfm.get_confusion_matrix()[1][1]/(cfm.get_confusion_matrix()[1][1]+cfm.get_confusion_matrix()[0][1])
    whole_las_metric_dict["recall"] = cfm.get_confusion_matrix()[1][1]/(cfm.get_confusion_matrix()[1][1]+cfm.get_confusion_matrix()[1][0])

    return model_las_metric_dict, whole_las_metric_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval entire folder.')
    parser.add_argument('folder', type=str, help='Folder with laz files to predict on')
    parser.add_argument('model', type=str, help='Path to the model')

    # args = parser.parse_args()
    # data_path = args.folder
    # model_path = args.model
    data_path = "/home/jf/data"
    raw_data_path = "/home/jf/data/denmark/raw/"
    model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-11/15-51-44/SEUNet50.pt"
    model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-07/13-32-19/SEUNet50.pt"


    model, transform_test, config = load_model(model_path, data_path)

    ## load transform pt pre
    processed_folder_name = config["processed_folder"] 
    data_root_path = os.path.join(config['dataroot'] , "denmark")
    processed_data_root_path = os.path.join(data_root_path, processed_folder_name)


    # total_points = 0
    # total_pl_points = 0
    # amount_of_files = 0
    # dirs = list(glob.glob(data_path+"*"))
    # for dir in dirs:
    #     files = list(glob.glob(dir+"/*.l*"))
    #     for file in files:
    #         data = laspy.read(file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    #         label_data = data[data.classification == 14]


    splits = ["train", "val", "test"]

    metric_dict = {}
    for split in splits:
        if split == "train":
            overlap = config["train_overlap"]
        if split == "val":
            overlap = 0
        if split == "test":
            overlap = 0
        predict_folder_name = f"{split}_{overlap}_({config['block_size_x']}, {config['block_size_y']})"
        predict_folder = os.path.join(processed_data_root_path, predict_folder_name)
        pre_trans_path = os.path.join(predict_folder, "stats.pt")
        room_info = torch.load(pre_trans_path)
        
        files = room_info['room_names'][:1]
        for filename in files:
            print(f"Predecting on {filename}")
            #the unprocessed file
            raw_file_path = os.path.join(raw_data_path,split,filename+".laz")
            raw_file = laspy.read(raw_file_path, laz_backend=laspy.compression.LazBackend.LazrsParallel)
            #the file that the models sees
            model_file_path = os.path.join(raw_data_path,split,"NewLaz",filename+".laz")
            model_file = laspy.read(model_file_path, laz_backend=laspy.compression.LazBackend.LazrsParallel)

            pred_data = predict(room_info, model, filename, transform_test, predict_folder)
            model_las_metric_dict, whole_las_metric_dict = get_metrics(pred_data, model_file,raw_file)
            metric_dict[filename] = {"model":model_las_metric_dict, "whole":whole_las_metric_dict}
    
    print(json.dumps(metric_dict,sort_keys=True, indent=4))

    with open("metric_dict.json", "w") as write_file:
        json.dump(metric_dict, write_file, indent=4)
