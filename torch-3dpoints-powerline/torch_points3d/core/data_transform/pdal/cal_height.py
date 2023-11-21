import argparse
from functools import partial
import pdal
from multiprocessing import Pool

from pathlib import Path

def worker(dir: Path, output_dir: Path, file: str):
    file_name = file.name
    out_file = file_name.split(".")[0]
    input_file = dir.joinpath(file)
    
    json = """
    [
        "%s",
        {
            "type":"filters.hag_nn"
        },
        {
            "type":"writers.las",
            "filename":"%s/%s_hag_nn.laz",
            "extra_dims":"HeightAboveGround=float32",
            "compression":"laszip"
        }
    ]
    """ % (input_file, str(output_dir), out_file)

    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    return 


def cal_height(dir: Path, output_dir: Path, MAX_WORKERS: int):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]

    func = partial(worker, dir, output_dir)

    with Pool(MAX_WORKERS) as p:
        p.map(func, onlyfiles)
    
    p.join()
    p.close()


    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the height from the ground.')
    parser.add_argument('folder', type=str, help='Folder with files to calculation')

    args = parser.parse_args()
    dir = args.folder

    height_dir_name = "LazFilesWithHeightParam"

    height_dir = Path(f"{dir}/{height_dir_name}")
    height_dir.mkdir(exist_ok=True)

    dir = Path(f"{dir}")
    cal_height(dir, height_dir, 3)