import pytest
import torch


def _build_subtomo_dir(root, native_size, crop_size, n_fitting, n_val, ctf_ones, double_size_ctf):
    """
    Builds a synthetic subtomo_dir with the layout SubtomoDataset/fit_model expect:
        {root}/{fitting,val}_subtomos/{subtomo0,subtomo1,ctf}/{idx}.pt
    'subtomo0'/'subtomo1' are at native_size; 'ctf' is at crop_size, in rfftn convention
    (crop_size, crop_size, crop_size // 2 + 1), or at twice crop_size if 'double_size_ctf'.
    """
    for split, n in [("fitting_subtomos", n_fitting), ("val_subtomos", n_val)]:
        for sub in ["subtomo0", "subtomo1", "ctf"]:
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for idx in range(n):
            s0 = torch.randn(native_size, native_size, native_size)
            s1 = s0 + 0.1 * torch.randn(native_size, native_size, native_size)
            torch.save(s0, root / split / "subtomo0" / f"{idx}.pt")
            torch.save(s1, root / split / "subtomo1" / f"{idx}.pt")
            ctf_size = 2 * crop_size if double_size_ctf else crop_size
            ctf_shape = (ctf_size, ctf_size, ctf_size // 2 + 1)
            if ctf_ones:
                ctf = torch.ones(ctf_shape)
            else:
                ctf = torch.rand(ctf_shape).clamp(0, 1)
            torch.save(ctf, root / split / "ctf" / f"{idx}.pt")


@pytest.fixture
def make_subtomo_dir(tmp_path):
    """
    Factory fixture. Call as:
        make_subtomo_dir(native_size, crop_size, n_fitting, n_val, ctf_ones=False, double_size_ctf=False)
    Returns the subtomo_dir root (a Path), containing 'fitting_subtomos'/'val_subtomos'.
    """

    def _make(native_size, crop_size, n_fitting, n_val, ctf_ones=False, double_size_ctf=False, name="subtomo_dir"):
        root = tmp_path / name
        _build_subtomo_dir(root, native_size, crop_size, n_fitting, n_val, ctf_ones, double_size_ctf)
        return root

    return _make
