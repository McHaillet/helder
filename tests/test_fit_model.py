"""
End-to-end tests for 'helder fit-model' on a per-subtomo 3D-CTF subtomo_dir (the only
supported mode - the legacy mw_angle/binary-wedge path has been removed entirely). These
call fit_model() directly (not through the CLI). Most of these actually run the U-Net,
which requires a GPU: fit_model always builds its pytorch_lightning Trainer with
accelerator="gpu".
"""
import shutil

import pytest
import torch

from helder.fit_model import fit_model

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fit_model always builds its Trainer with accelerator='gpu'",
)


def _fit_kwargs(subtomo_dir, logdir, crop_size, num_downsample_layers=3, lambda_=2.0):
    return dict(
        unet_params_dict={"chans": 4, "num_downsample_layers": num_downsample_layers, "drop_prob": 0.0},
        adam_params_dict={"lr": 1e-3},
        num_epochs=2,
        batch_size=2,
        subtomo_size=crop_size,
        gpu=[0],
        num_workers=2,
        subtomo_dir=str(subtomo_dir),
        logdir=str(logdir),
        logger="csv",
        check_val_every_n_epochs=1,
        lambda_=lambda_,
        save_model_every_n_epochs=100,
        save_n_models_with_lowest_fitting_loss=0,
        save_n_models_with_lowest_val_loss=0,
        seed=0,
    )


def test_fit_model_requires_ctf_dir(make_subtomo_dir, tmp_path):
    # raises before the Trainer/GPU is ever touched, so no GPU needed here
    root = make_subtomo_dir(native_size=24, crop_size=24, n_fitting=4, n_val=0)
    shutil.rmtree(root / "fitting_subtomos" / "ctf")
    with pytest.raises(ValueError, match="ctf"):
        fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=24))


@requires_gpu
def test_fit_model_completes(make_subtomo_dir, tmp_path):
    root = make_subtomo_dir(native_size=24, crop_size=24, n_fitting=6, n_val=2)
    fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=24))


@requires_gpu
def test_fit_model_raises_on_indivisible_native_size(make_subtomo_dir, tmp_path):
    # native/on-disk size (30) not divisible by 2**num_downsample_layers (8): the model is
    # run on native-size subtomos every step, so this must raise rather than silently pad
    root = make_subtomo_dir(native_size=30, crop_size=24, n_fitting=6, n_val=2)
    with pytest.raises(ValueError, match="divisible"):
        fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=24))


def test_fit_model_raises_when_native_size_not_equal_to_subtomo_size(make_subtomo_dir, tmp_path):
    # raises before the Trainer/GPU is ever touched, so no GPU needed here. The model is run
    # directly on the on-disk subtomo0/subtomo1 every step, rotating its own estimate in
    # place with an exact, shape-preserving grid rotation (see helder.utils.rotation), so the
    # on-disk size must equal subtomo_size - no cropping happens anymore.
    root = make_subtomo_dir(native_size=32, crop_size=24, n_fitting=6, n_val=2)
    with pytest.raises(ValueError, match="equal"):
        fit_model(**_fit_kwargs(root, tmp_path / "logs", crop_size=24))
