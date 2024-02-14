from pathlib import Path
import numpy as np
from volumetric_sim_tools.DataUtils.AnatomicalVolume import AnatomicalVolume
from volumetric_sim_tools.DataUtils.DataMerger import DataMerger
import click
from pairwise_dice import rm_voxel

@click.command()
@click.option("--vol_path", required=True, help="path to first volume pngs")
@click.option("--rec_path", required=True, help="path to trials")

def check(vol_path, rec_path):
    vol_path = Path(vol_path)
    vol = AnatomicalVolume.from_png_list(vol_path)
    rec_path = Path(rec_path)

    rm_voxel(vol, rec_path)

def main():
    check()

if __name__ == "__main__":
    main()