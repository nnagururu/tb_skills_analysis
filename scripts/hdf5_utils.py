import h5py
import numpy as np

def save_dict_to_hdf5(data, filename):
    """
    Save a dictionary to an HDF5 file.

    Parameters:
    - data (dict): Dictionary where keys are strings and values are numpy arrays.
    - filename (str): Filename for the HDF5 file.
    """
    with h5py.File(filename, "w") as hdf:
        for key, value in data.items():
            hdf.create_dataset(key, data=value)

def load_dict_from_hdf5(filename):
    """
    Load an HDF5 file into a dictionary.

    Parameters:
    - filename (str): Filename of the HDF5 file to read.

    Returns:
    - dict: Dictionary with keys and values loaded from the HDF5 file.
    """
    data = {}
    with h5py.File(filename, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][:]
    return data

def save_dict_to_hdf5_recursive(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Take a dictionary `dic` and save its contents to the HDF5 group `path` within the HDF5 file `h5file`.
    """
    for key, item in dic.items():
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            h5file[path + key] = item

def load_dict_from_hdf5_recursive(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
        else:
            ans[key] = item[()]
    return ans