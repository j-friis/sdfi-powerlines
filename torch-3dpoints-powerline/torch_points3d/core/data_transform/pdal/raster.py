import argparse
import pdal
#from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from functools import partial

def worker(output_dir: Path, resolution: str, file: str):
    input_file = file.name
    out_file = input_file.replace("_height_filtered",'')
    out_file = out_file.split(".")[0]

    json = """
    [
        "%s",   
        {
            "type":"writers.gdal",
            "filename":"%s/%s_max.tif",
            "output_type":"max",
            "gdaldriver":"GTiff",
            "resolution":%s
        }
    ]
    """ % (str(file), str(output_dir), out_file, resolution)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    return


def rasterize(dir: Path, output_dir: Path, resolution: str, MAX_WORKERS: int):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]
    func = partial(worker, output_dir, resolution)

    with Pool(MAX_WORKERS) as p:
        p.map(func, onlyfiles)
    p.join()
    p.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize the laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to convert')
    parser.add_argument('Resolution', type=str, help='The raster resolution')

    args = parser.parse_args()
    dir = args.folder
    resolution = args.Resolution
    height_removed_dir_name = "LazFilesWithHeightRemoved"
    height_removed_dir = Path(f"{dir}/{height_removed_dir_name}")
    height_removed_dir.mkdir(exist_ok=True)

    if resolution == "1.0":
        raster_image_dir_name  = "ImagesGroundRemoved"
        raster_image_dir = Path(f"{dir}/{raster_image_dir_name}")
        raster_image_dir.mkdir(exist_ok=True)
    else:
        raster_image_dir_name  = "ImagesGroundRemovedLarge"
        raster_image_dir = Path(f"{dir}/{raster_image_dir_name}")
        raster_image_dir.mkdir(exist_ok=True)

    rasterize(height_removed_dir, raster_image_dir, resolution, 1)
