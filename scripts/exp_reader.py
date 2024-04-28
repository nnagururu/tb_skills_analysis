from pathlib import Path
from collections import OrderedDict
from natsort import natsorted
import numpy as np
import h5py
import pandas as pd

class ExpReader:
    def __init__(self, recording_path, ignore_keys = ['depth', 'r_img', 'l_img', 'segm'], verbose = False):
        self.recording_path = Path(recording_path)
        self.hdf5_files = []
        self._data = OrderedDict()
        self.ignore_keys = ignore_keys

        self._get_merged_data(verbose = verbose)
        self.extract_from_data()

    def _clear_data(self):
        for g in self._data.keys():
            self._data[g].clear()

        self.hdf5_files = []
    
    def _get_merged_data(self, verbose):
        self._clear_data()
        print(self.recording_path)
        
        self.hdf5_files = list(self.recording_path.glob('*.hdf5'))
        self.hdf5_files = natsorted(self.hdf5_files)
        print("Number of Files ", len(self.hdf5_files))

        for idx, file_name in enumerate(self.hdf5_files):
            try:
                file = h5py.File(file_name, "r")
            except OSError as e:
                print(f"Error opening file {file_name}: {e}")
                if verbose:
                    print(f"Skipping file {file_name} due to error.")
                continue  # Skip to the next file
            
            
            if verbose:
                print(idx, "Opening", file_name)
            for grp in file.keys():
                if grp == "metadata":
                    continue
                if grp not in self._data:
                    self._data[grp] = OrderedDict()
                if verbose:
                    print("\t Processing Group ", grp)
                
                # Handling voxels removed independently because index recorded in voxel_color
                # is reset with each hdf5 file
                if grp == "voxels_removed":
                    if 'voxel_time_stamp' not in file[grp].keys() or len(file[grp]['voxel_time_stamp']) == 0:
                            continue
                    
                    if verbose:
                        print("\t\t Processing Dataset ", 'voxel_time_stamp')
                        print("\t\t Processing Dataset ", 'voxel_color')
                        print("\t\t Processing Dataset ", 'voxel_removed')

                    if 'voxel_time_stamp' not in self._data[grp]:
                        self._data[grp]['voxel_time_stamp'] = file[grp]['voxel_time_stamp'][()]
                        self._data[grp]['voxel_color'] = file[grp]['voxel_color'][()]
                        self._data[grp]['voxel_removed'] = file[grp]['voxel_removed'][()]
                    else:
                        max_ts_index = len(self._data[grp]['voxel_time_stamp'])
                        curr_vox_color = file[grp]['voxel_color'][()]
                        curr_vox_removed = file[grp]['voxel_removed'][()]
                        curr_vox_color[:,0] += max_ts_index
                        curr_vox_removed[:,0] += max_ts_index
                        
                        self._data[grp]['voxel_color'] = np.append(
                            self._data[grp]['voxel_color'], curr_vox_color, axis = 0
                        )
                        self._data[grp]['voxel_removed'] = np.append(
                            self._data[grp]['voxel_removed'], curr_vox_removed, axis = 0
                        )
                        self._data[grp]['voxel_time_stamp'] = np.append(
                            self._data[grp]['voxel_time_stamp'], file[grp]['voxel_time_stamp'][()]
                        )
                    continue


                for dset in file[grp].keys():
                    # print(dset)
                    if grp == "data" and (dset in self.ignore_keys):
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
    
    def extract_from_data(self):
        self.cam_poses = self._data['data']['pose_main_camera']
        self.vol_poses = self._data['data']['pose_mastoidectomy_volume']
        self.d_poses = self._data['data']['pose_mastoidectomy_drill']
        self.data_ts = self._data['data']['time']

        if 'drill_force_feedback' in self._data:
            self.forces = self._data['drill_force_feedback']['wrench']
            self.forces_ts = self._data['drill_force_feedback']['time_stamp']
        
        self.v_rm_ts = self._data['voxels_removed']['voxel_time_stamp']
        self.v_rm_colors = self._data['voxels_removed']['voxel_color']
        self.v_rm_locs = self._data['voxels_removed']['voxel_removed']

        if 'burr_size' in self._data['burr_change']:
            self.burr_chg_sz = self._data['burr_change']['burr_size']
            self.burr_chg_ts = self._data['burr_change']['time_stamp']
        else:
            self.burr_chg_sz = None
            self.burr_chg_ts = None

        