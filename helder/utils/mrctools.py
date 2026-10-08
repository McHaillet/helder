import os
import shutil

import mrcfile
import torch


def load_mrc_data(mrc_file):
    """
    Loads a .mrc or .rec file as a torch tensors.
    """
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        try:
            data = torch.tensor(mrc.data)
        except TypeError:
            data = torch.tensor(mrc.data.astype(float))
    return data


def normalize_tomo(tomo):
    """
    Subtracts the mean and divides by the median absolute deviation (MAD), scaled by 1.4826 so that it estimates the standard deviation for normally distributed data.
    """
    tomo = tomo - tomo.mean()
    mad = (tomo - tomo.median()).abs().median()
    return tomo / (1.4826 * mad)


def save_mrc_data(data, mrc_file, save=False):
    """
    Saves a torch tensor as an .mrc file.
    """
    if save:
        if os.path.exists(mrc_file):
            print(f"File '{mrc_file}' already exists! Moving it to '{mrc_file}~'")
            shutil.move(mrc_file, f"{mrc_file}~")
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(data.numpy())
