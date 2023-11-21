import argparse
import os
from filter_height import filter_height
from raster import rasterize
from cal_height import cal_height
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run entire pipeline.')
    parser.add_argument('folder', type=str, help='Folder with files to run pipeline on')
    parser.add_argument('max_workers', type=int, help='The number of process used in Pool')
    parser.add_argument('height_filter', type=float, help='The amount of meters above ground we remove')
    parser.add_argument('Resolution', type=str, help='The raster resolution')


    args = parser.parse_args()
    dir = args.folder
    dir = dir.replace('~',os.path.expanduser('~'))
     
    MAX_WORKERS = args.max_workers
    height_filter = args.height_filter
    resolution = args.Resolution


    height_dir_name = "LazFilesWithHeightParam"
    height_removed_dir_name = "LazFilesWithHeightRemoved"
    height_dir = Path(f"{dir}/{height_dir_name}")
    height_removed_dir = Path(f"{dir}/{height_removed_dir_name}")

    if resolution == "1.0":
        raster_image_dir_name  = "ImagesGroundRemoved"
        raster_image_dir = Path(f"{dir}/{raster_image_dir_name}")
        raster_image_dir.mkdir(exist_ok=True)
    else:
        raster_image_dir_name  = "ImagesGroundRemovedLarge"
        raster_image_dir = Path(f"{dir}/{raster_image_dir_name}")
        raster_image_dir.mkdir(exist_ok=True)

    dir = Path(f"{dir}")

    raster_image_dir.mkdir(exist_ok=True)
    height_dir.mkdir(exist_ok=True)
    height_removed_dir.mkdir(exist_ok=True)

    print("------------------- Calculating height -------------------")
    cal_height(dir, height_dir, MAX_WORKERS)
    laz_with_height_dir = f"{dir}/LazFilesWithHeightParam"
    print("-------------------- Filtering height --------------------")
    filter_height(height_dir, height_removed_dir, MAX_WORKERS, height_filter)
    print("-------------------- Rasterize files ---------------------")
    laz_with_height_removed_dir = f"{dir}/LazFilesWithHeightRemoved"
    rasterize(height_removed_dir, raster_image_dir, resolution, MAX_WORKERS)
