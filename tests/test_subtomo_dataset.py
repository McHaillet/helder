"""
Tests for SubtomoDataset: it now just loads raw (subtomo0, subtomo1, ctf) triples - no
rotation, no masking, no legacy mw_angle path (all of that moved into LitUnet3D._step, see
helder.utils.unet). 'ctf' is loaded as-is, in rfftn convention (never expanded to a full,
fftshifted array - see apply_fourier_mask_to_tomo). Pure CPU/torch - no GPU needed.
"""
import pytest
import torch

from helder.utils.subtomo_dataset import SubtomoDataset


def test_missing_subtomo0_raises(tmp_path):
    root = tmp_path / "subtomo_dir" / "fitting_subtomos"
    (root / "subtomo1").mkdir(parents=True)
    (root / "ctf").mkdir(parents=True)
    with pytest.raises(ValueError, match="subtomo0"):
        SubtomoDataset(subtomo_dir=str(root))


def test_missing_ctf_raises(tmp_path):
    root = tmp_path / "subtomo_dir" / "fitting_subtomos"
    (root / "subtomo0").mkdir(parents=True)
    (root / "subtomo1").mkdir(parents=True)
    with pytest.raises(ValueError, match="ctf"):
        SubtomoDataset(subtomo_dir=str(root))


def test_item_shapes_and_value_range(make_subtomo_dir):
    native, crop = 32, 24
    root = make_subtomo_dir(native_size=native, crop_size=crop, n_fitting=4, n_val=2)
    ds = SubtomoDataset(subtomo_dir=str(root / "fitting_subtomos"))
    item = ds[0]
    assert item["subtomo0"].shape == (native, native, native)
    assert item["subtomo1"].shape == (native, native, native)
    assert item["ctf"].shape == (crop, crop, crop // 2 + 1)
    assert item["ctf"].min() >= -1e-4
    assert item["ctf"].max() <= 1 + 1e-4
    # continuous, not a binary wedge
    assert not torch.all((item["ctf"] == 0) | (item["ctf"] == 1))
    assert item["index"] == 0


def test_ctf_dc_bin_forced_to_one(make_subtomo_dir):
    """
    The DC bin's on-disk value is the (small) physical CTF amplitude, but __getitem__ must
    override it to 1 so data_consistency_loss gives it full weight - see SubtomoDataset's
    docstring.
    """
    native, crop = 32, 24
    root = make_subtomo_dir(native_size=native, crop_size=crop, n_fitting=4, n_val=0)
    ds = SubtomoDataset(subtomo_dir=str(root / "fitting_subtomos"))
    for i in range(len(ds)):
        assert ds[i]["ctf"][0, 0, 0] == 1.0


def test_len_ignores_stray_non_pt_files(make_subtomo_dir):
    """
    A stray non-'.pt' file in subtomo0/ (a hidden dotfile, an NFS silly-rename artifact, a
    leftover from an interrupted run, ...) must not inflate __len__ past the actual number
    of samples - otherwise a sampler would eventually request an index that doesn't exist
    on disk.
    """
    native, crop = 32, 24
    root = make_subtomo_dir(native_size=native, crop_size=crop, n_fitting=4, n_val=0)
    subtomo0_dir = root / "fitting_subtomos" / "subtomo0"
    (subtomo0_dir / ".nfs0000000012345678").write_bytes(b"")
    (subtomo0_dir / ".DS_Store").write_bytes(b"")

    ds = SubtomoDataset(subtomo_dir=str(root / "fitting_subtomos"))
    assert len(ds) == 4
