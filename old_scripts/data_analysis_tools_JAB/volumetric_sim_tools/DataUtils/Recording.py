from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import h5py
import pandas as pd
from natsort import natsorted


@dataclass
class Recording:
    """Class storing all the HDF5 files from a experiment"""

    data_dir: Path
    participant_id: str = None
    anatomy: str = None
    guidance_modality: str = None
    trial_idx: int = None

    def __post_init__(self):
        if not self.data_dir.exists():
            print("provided path does not exists")
            exit(0)

        # Store paths in self.file_list
        # Store hdf5 handlers in self.data_dict
        self.file_list: List[Path] = None
        self.data_dict: Dict[int, h5py.File] = None
        self.file_list, self.data_dict = self.load_hdf5()

    def load_hdf5(self) -> OrderedDict:
        """Create a dictionary with all the hdf5 files sorted by name.
           Load only files that contain at least one voxel removed.

        Returns
        -------
        OrderedDict

        """
        files_list = []
        data_dict = OrderedDict()
        for file in self.data_dir.glob("*.hdf5"):
            files_list.append(file)

        files_list = natsorted(files_list)  # sort files by name
        final_file_list = []
        idx = 0
        for file in files_list:
            try:
                h5py_file = h5py.File(file, "r")
                # Only add files that have at least one voxel removed
                if "voxels_removed/voxel_time_stamp" in h5py_file:
                    final_file_list.append(file)
                    data_dict[idx] = h5py_file
                    idx += 1
            except:
                pass

        return final_file_list, data_dict

    def close_files(self):
        v: h5py.File
        for k, v in self.data_dict.items():
            v.close()

    def __len__(self):
        return len(self.file_list)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close_files()

    def concatenate_data(self, dataset_name: str):
        result = []
        for k, v in self.data_dict.items():
            if dataset_name in v:
                result.append(v[dataset_name])

        if len(result) > 0:
            result = np.concatenate(result) if len(result[0].shape) == 1 else np.vstack(result)

        return result

    def count_frames(self) -> int:
        frames_count = 0
        for k, file in self.data_dict.items():
            frames_count += file["data"]["time"].shape[0]
        return frames_count

    def img_data_iterator(self):
        """Iterates over all the image data of a Recording

        Yields
        ------
        Tuple
            returns (l_img[idx], r_img[idx], depth[idx], segm[idx])
        """
        for k, file in self.data_dict.items():
            l_img = file["data"]["l_img"]
            r_img = file["data"]["r_img"]
            depth = file["data"]["depth"]
            segm = file["data"]["segm"]

            for idx in range(l_img.shape[0]):
                yield (l_img[idx], r_img[idx], depth[idx], segm[idx])


if __name__ == "__main__":
    # path = Path(
    #     "/home/juan1995/research_juan/cisII_SDF_project/Data/PilotData/sdf_daniel_pilot_guidance/"
    # )
    # meta_data = dict(participant_id="participant1", anatomy="A", guidance_modality="Baseline")

    # with Recording(path, **meta_data) as recording:
    #     print(f"Read {len(recording)} h5 files for {recording.participant_id}")

    import json
    root_path = Path("/home/juan1995/research_juan/cisII_SDF_project/Data/UserStudy2_IROS/")
    p1 = root_path / "Participant_08/2023-02-08 10:07:14_AnatomyA_baseline"
    p2 = root_path / 'Participant_08/2023-02-13 09:39:29_AnatomyJ_Force'

    path = p2
    with open(path/"meta.json","r") as f:
        meta_data = json.load(f)
    # meta_data = dict(participant_id="participant8", anatomy="J", guidance_modality="Haptic")

    with Recording(path, **meta_data) as recording:
        print(f"Read {len(recording)} h5 files for {recording.participant_id}")