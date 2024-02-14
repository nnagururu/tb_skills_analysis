from pathlib import Path
from collections import OrderedDict
from natsort import natsorted
import numpy as np
import h5py
import pandas as pd

class exp_reader:
    def __init__(self, recording_path):
        self.recording_path = recording_path
        self.hdf5_files = []
        self._data = OrderedDict()

        self.get_merged_data()

    def _clear_data(self):
        for g in self._data.keys():
            self._data[g].clear()

        self.hdf5_files = []
    
    def get_merged_data(self, verbose = False):
        self._clear_data()
        
        self.hdf5_files = list(self.recording_path.glob('*.hdf5'))
        self.hdf5_files = natsorted(self.hdf5_files)
        print("Number of Files ", len(self.hdf5_files))

        for idx, file_name in enumerate(self.hdf5_files):
            file = h5py.File(file_name, "r")
            if verbose:
                print(idx, "Opening", file_name)
            for grp in file.keys():
                if grp == "metadata":
                    continue
                if grp not in self._data:
                    self._data[grp] = OrderedDict()
                if verbose:
                    print("\t Processing Group ", grp)
                for dset in file[grp].keys():
                    # print(dset)
                    if grp == "data" and (dset == "depth" or dset == "r_img" or dset == "pose_mastoidectomy_volume"):
                        continue

                    if len(file[grp][dset]) == 0:
                        continue

                    if verbose:
                        print("\t\t Processing Dataset ", dset)
                    if dset not in self._data[grp]:
                        self._data[grp][dset] = file[grp][dset][()]
                    else:
                        self._data[grp][dset] = np.append(
                            self._data[grp][dset], file[grp][dset][()], axis=0
                        )
            file.close()
        
        return self._data 