import glob
import os

import time
import torch
from torch.utils.data import Dataset


def safe_load(file_path, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return torch.load(file_path)
        # except everything to catch all exceptions
        except Exception as e:
            print(f"Error loading {file_path}")
            if attempt == max_retries - 1:
                raise e  # Reraise if it's the last attempt
            print(f"Error message is: {e}")
            print(f"Retrying in {delay} seconds")
            time.sleep(delay)  # Wait before retrying


class SubtomoDataset(Dataset):
    """
    A torch dataset producing raw (subtomo0, subtomo1, ctf) triples used for model fitting.
    'subtomo_dir' must have 'subtomo0/', 'subtomo1/' and 'ctf/' subdirectories, each holding
    matching '{index}.pt' files.

    'subtomo0'/'subtomo1' are two independent-noise reconstructions of the same tomogram
    region (e.g. from even/odd tilt series frames), at 'subtomo_size' - the model is run on
    them directly, with no cropping: LitUnet3D rotates its own estimate using one of the 20
    grid-aligned rotations from helder.utils.rotation.get_grid_rotations, which is exact (no
    interpolation) and preserves shape, so no extra border is needed to rotate into. Both
    share the same physical 'ctf' (values in [0, 1]), since it depends only on the
    acquisition geometry, not on which half of the frames was used. The DC bin ([0,0,0]) is
    forced to 1 on load, overriding its (small) physical value - see __getitem__.

    'ctf/{index}.pt' must be given at the same 'subtomo_size' as subtomo0/subtomo1: a CTF,
    unlike a binary missing-wedge mask, has genuine radial/magnitude frequency dependence and
    can't be correctly resized in Fourier space, so it must be independently, correctly
    reconstructed at that resolution. Each tensor must be real-valued, in rfftn convention:
    shape (N, N, N//2+1), DC at index [0,0,0] (unshifted) - matching what
    apply_fourier_mask_to_tomo expects. Unlike a real-space volume, this mask is never
    rotated anywhere in this codebase, so there is no need to convert it to a full,
    fftshifted (N, N, N) array (see apply_fourier_mask_to_tomo).
    """

    def __init__(self, subtomo_dir):
        super().__init__()
        self.subtomo_dir = subtomo_dir
        for sub in ["subtomo0", "subtomo1", "ctf"]:
            if not os.path.isdir(f"{subtomo_dir}/{sub}"):
                raise ValueError(f"'{subtomo_dir}' must contain a '{sub}' subdirectory.")

    def __len__(self):
        # count only "*.pt" files, not every directory entry: a stray non-'.pt' file (a
        # hidden dotfile, an NFS silly-rename artifact from a deleted-while-open file, a
        # leftover from an interrupted run, ...) would otherwise inflate the count past the
        # number of actual, contiguously-indexed samples, causing __getitem__ to be asked
        # for an index one-past-the-end that doesn't exist on disk
        return len(glob.glob(f"{self.subtomo_dir}/subtomo0/*.pt"))

    def __getitem__(self, index):
        subtomo0_file = f"{self.subtomo_dir}/subtomo0/{index}.pt"
        subtomo1_file = f"{self.subtomo_dir}/subtomo1/{index}.pt"
        subtomo0 = safe_load(subtomo0_file)
        subtomo1 = safe_load(subtomo1_file)
        ctf = safe_load(f"{self.subtomo_dir}/ctf/{index}.pt")
        # force full weight at the DC bin: its physical CTF value is only the (tiny)
        # amplitude-contrast fraction, which would otherwise leave the model's overall
        # scale/mean almost unconstrained by data_consistency_loss
        ctf[0, 0, 0] = 1.0
        return {
            "subtomo0": subtomo0,
            "subtomo1": subtomo1,
            "ctf": ctf,
            "subtomo0_file": subtomo0_file,
            "subtomo1_file": subtomo1_file,
            "index": index,
        }


# %%
