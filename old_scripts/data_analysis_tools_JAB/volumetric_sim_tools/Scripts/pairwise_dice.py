from pathlib import Path
import numpy as np
from volumetric_sim_tools.DataUtils.AnatomicalVolume import AnatomicalVolume
from volumetric_sim_tools.DataUtils.DataMerger import DataMerger
import click
from itertools import combinations



def check_path(path: Path):
    if not path.exists():
        print(f"Path {path} does not exist")
        exit(0)

    return path

def rm_voxel(vol, hdf5_dir):
    experiment_data = DataMerger()
    experiment_data.get_merged_data(hdf5_dir)
    removed_voxels = experiment_data.get_removed_voxels()
    vol.remove_voxels(removed_voxels)

    return vol

def binarize_removed(vol):
    vol = vol.reshape(-1,4)
    vol_bin = []
    for i in vol:
        if np.array_equal(i, [0., 0., 0., 0.]):
            continue
        if np.array_equal(i, [255., 0., 0., 50.]):
            vol_bin.append(True)
        else:
            vol_bin.append(False)

    return np.asarray(vol_bin)

            

def dice(vol_path: str, rec1_path: str, rec2_path: str):
    vol_path = check_path(Path(vol_path))
    rec1_path = check_path(Path(rec1_path))
    rec2_path = check_path(Path(rec2_path))

    vol1 = AnatomicalVolume.from_png_list(vol_path)
    vol2 = AnatomicalVolume.from_png_list(vol_path)

    vol1  = rm_voxel(vol1, rec1_path).anatomy_matrix
    vol2  = rm_voxel(vol2, rec2_path).anatomy_matrix
    
    vol1_b = binarize_removed(vol1)
    vol2_b = binarize_removed(vol2)

    intersection = np.logical_and(vol1_b, vol2_b)
    dice = 2. * intersection.sum()/(vol1_b.sum() + vol2_b.sum())

    return dice

@click.command()
@click.option("--vol_path", required=True, help="path to first volume pngs")
@click.option("--rec_path", required=True, help="path to trials")

def pairwise_dice(vol_path: str, rec_path: str):
    rec_path = Path(rec_path)
    folder_paths = [str(sub) for sub in rec_path.iterdir() if sub.is_dir()]
    pairs = list(combinations(folder_paths, 2))
    results = []
    for pair in pairs:
        print(pair[0])
        print(pair[1])
        results.append(dice(vol_path, pair[0], pair[1]))
    print(results)
    print(np.mean(results))


def main():
    pairwise_dice()


if __name__ == "__main__":
    main()