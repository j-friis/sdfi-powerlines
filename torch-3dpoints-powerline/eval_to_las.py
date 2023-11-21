import argparse
from pathlib import Path
import glob
import os
from timeit import default_timer as timer

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


def load_model(model_path: str, data_path: str, model_weight: str, path_to_cnn: str):
    model = torch.load(model_path)
    model['run_config']['data']['dataroot'] = data_path
    try:
        model['run_config']['data']['path_to_model'] = path_to_cnn
    except:
        pass
    torch.save(model, model_path)
    print(model['run_config']["data"])


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
    model_pl = PretainedRegistry.from_file(model_path, weight_name=model_weight).cuda()
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

def add_predection_to_laz_files(filename, data_root_path, pred_data, processed_folder_name):
    ## read original las file
    normal_laz_file = os.path.join(data_root_path, "raw", "test", filename+".laz")

    non_processed_laz = laspy.read(normal_laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    non_processed_point_data = np.stack([non_processed_laz.X, non_processed_laz.Y, non_processed_laz.Z], axis=0).transpose((1, 0))

    powerline_pts = pred_data[np.where(pred_data[:,3] == 1)].copy()
    powerline_pts_coord = powerline_pts[:,:-1].astype(np.int32) 

    non_processed_laz.add_extra_dim(laspy.ExtraBytesParams(
        name="prediction",
        type=np.uint8,
        description="The prediction of the model"
        ))

    idx = get_nearest_neighbors(powerline_pts_coord, non_processed_point_data)
    pred = np.zeros(len(non_processed_point_data))
    pred[idx] = 1
    non_processed_laz.prediction = pred

    processed_data_root_path = os.path.join(data_root_path, processed_folder_name)
    eval_folder = data_root_path + "/eval" 
    Path(eval_folder).mkdir(exist_ok=True, parents=True)
    eval_file_name = os.path.join(eval_folder, filename+".laz")
    
    non_processed_laz.write(str(eval_file_name), do_compress = True, laz_backend=laspy.compression.LazBackend.LazrsParallel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval entire folder.')
    parser.add_argument('folder', type=str, help='Folder with laz files to predict on')
    parser.add_argument('model', type=str, help='Path to the model')
    parser.add_argument('model_metric', type=str, help='What metric to chose the model from options are latest, loss_seg, acc, macc and miou')
    parser.add_argument('multi_view_model', type=str, help='The model that segments in 2D')

    args = parser.parse_args()
    data_path = args.folder
    model_path = args.model
    model_metric = args.model_metric
    multi_view_model = args.multi_view_model

    data_path = data_path.replace('~',os.path.expanduser('~'))
    model_path = model_path.replace('~',os.path.expanduser('~'))
    multi_view_model = multi_view_model.replace('~',os.path.expanduser('~'))

    # data_path = "/home/jf/data"
    # # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-04-18/12-08-50/SEUNet18.pt"
    # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-02/16-01-32/SEUNet18.pt"
    # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-02/11-26-09/SEUNet18.pt"
    # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-13/21-54-44/SEUNet18.pt"
    # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-13/10-49-02/SEUNet50.pt"
    # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-14/02-16-29/SEUNet50.pt"
    # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-13/21-54-44/SEUNet18.pt"
    # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-05-22/20-40-06/SEUNet18.pt"

    model, transform_test, config = load_model(model_path, data_path, model_metric, multi_view_model)

    ## load transform pt pre
    processed_folder_name = config["processed_folder"] 
    data_root_path = os.path.join(config['dataroot'] , "denmark")
    processed_data_root_path = os.path.join(data_root_path, processed_folder_name)
    predict_folder_name = f"test_0_({config['block_size_x']}, {config['block_size_y']})"
    predict_folder = os.path.join(processed_data_root_path, predict_folder_name)
    pre_trans_path = os.path.join(predict_folder, "stats.pt")
    room_info = torch.load(pre_trans_path)
    start = timer()
    for filename in room_info['room_names']:
        print(f"Predecting on {filename}")
        pred_data = predict(room_info, model, filename, transform_test, predict_folder)
        add_predection_to_laz_files(filename, data_root_path, pred_data, processed_folder_name)
    end = timer()
    print(end - start)