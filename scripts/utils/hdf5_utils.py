"""
hdf5_utils.py

This module provides functions for saving and loading dictionaries to/from HDF5 files.

Author: Nimesh Nagururu
"""

import h5py

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

def save_dict_to_hdf5_recursive(data, filename):
    """
    Recursively save a nested dictionary to an HDF5 file.

    Parameters:
    - data (dict): Dictionary to be saved.
    - filename (str): Filename for the HDF5 file.
    """
    with h5py.File(filename, 'w') as hdf:
        recursively_save_dict_contents_to_group(hdf, '/', data)

def recursively_save_dict_contents_to_group(hdf, path, data):
    """
    Recursively save dictionary contents to an HDF5 group.

    Parameters:
    - hdf (h5py.File): HDF5 file object.
    - path (str): Path within the HDF5 file.
    - data (dict): Dictionary to be saved.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            recursively_save_dict_contents_to_group(hdf, path + key + '/', value)
        else:
            hdf[path + key] = value

def load_dict_from_hdf5_recursive(filename):
    """
    Recursively load a nested dictionary from an HDF5 file.

    Parameters:
    - filename (str): Filename of the HDF5 file to read.

    Returns:
    - dict: Nested dictionary loaded from the HDF5 file.
    """
    with h5py.File(filename, 'r') as hdf:
        return recursively_load_dict_contents_from_group(hdf, '/')

def recursively_load_dict_contents_from_group(hdf, path):
    """
    Recursively load dictionary contents from an HDF5 group.

    Parameters:
    - hdf (h5py.File): HDF5 file object.
    - path (str): Path within the HDF5 file.

    Returns:
    - dict: Dictionary loaded from the HDF5 group.
    """
    data = {}
    for key, value in hdf[path].items():
        if isinstance(value, h5py.Group):
            data[key] = recursively_load_dict_contents_from_group(hdf, path + key + '/')
        else:
            data[key] = value[()]
    return data
