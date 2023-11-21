import argparse
import laspy
from multiprocessing import Pool
from pathlib import Path
from functools import partial

def worker(output_dir: Path, height_filter: int, file: str):
    file_name = file.name
    out_filename = file_name.split(".")[0]
    out_filename = out_filename.replace("_hag_delaunay",'')
    out_filename = out_filename.replace("_hag_nn",'')
    out_filename = f"{out_filename}_height_filtered.laz"
    out_file = output_dir.joinpath(out_filename)

    pdal_data = laspy.read(file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    pdal_data = pdal_data[pdal_data.HeightAboveGround>height_filter ]
    pdal_data.write(out_file, do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    return



def filter_height(dir: Path, output_dir: Path, MAX_WORKERS: int, height_filter: float):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]
    func = partial(worker, output_dir, height_filter)

    with Pool(MAX_WORKERS) as p:
        p.map(func, onlyfiles)

    p.join()
    p.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter the height in laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to filter')
    parser.add_argument('Height', type=float, help='Meters from the bottom to filter out')

    args = parser.parse_args()
    dir = args.folder
    height_filter = args.Height

    height_dir_name = "LazFilesWithHeightParam"
    height_dir = Path(f"{dir}/{height_dir_name}")
    height_dir.mkdir(exist_ok=True)

    height_removed_dir_name = "LazFilesWithHeightRemoved"
    height_removed_dir = Path(f"{dir}/{height_removed_dir_name}")
    height_removed_dir.mkdir(exist_ok=True)

    dir = Path(f"{dir}")
    filter_height(height_dir, height_removed_dir, 1, height_filter)
