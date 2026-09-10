"""
Tests for LitUnet3D's rotation helpers (helder.utils.unet): _sample_rotations/_rotate_batch.
Pure CPU/torch - no GPU needed (only fit_model's Trainer requires one).
"""
import random

import torch

from helder.utils.unet import LitUnet3D


def _make_lit_unet(**kwargs):
    return LitUnet3D(
        unet_params=dict(chans=2, num_downsample_layers=1),
        adam_params=dict(lr=1e-3),
        subtomo_size=8,
        **kwargs,
    )


def test_rotate_batch_round_trips_with_shared_rot_mats_when_deterministic():
    torch.manual_seed(0)
    lit_unet = _make_lit_unet()
    vol = torch.randn(3, 6, 6, 6)
    indices = [0, 5, 12]

    rot_mats = lit_unet._sample_rotations(indices, deterministic=True)
    rotated = lit_unet._rotate_batch(vol, rot_mats)
    back = lit_unet._rotate_batch(rotated, rot_mats, inverse=True)
    assert torch.allclose(back, vol)


def test_rotate_batch_round_trips_with_shared_rot_mats_when_non_deterministic():
    """
    Regression test: sample_grid_rotation(index, deterministic=False) draws from the shared
    global 'random' state, so two independent calls with the same 'index' are not guaranteed
    to agree - _sample_rotations must be called once per step and its result reused for both
    the forward and inverse _rotate_batch call, not re-derived from 'indices' each time.
    Advance the global random state between the two calls to make sure this is actually
    exercised.
    """
    torch.manual_seed(0)
    random.seed(1)
    lit_unet = _make_lit_unet()
    vol = torch.randn(3, 6, 6, 6)
    indices = [0, 5, 12]

    rot_mats = lit_unet._sample_rotations(indices, deterministic=False)
    rotated = lit_unet._rotate_batch(vol, rot_mats)
    random.random()  # perturb the shared global random state before the "inverse" call
    back = lit_unet._rotate_batch(rotated, rot_mats, inverse=True)
    assert torch.allclose(back, vol)


def test_step_combines_dc_and_eq_losses_weighted_by_lambda():
    torch.manual_seed(0)
    lit_unet = _make_lit_unet(lambda_=2.0)
    N = 8
    batch = {
        "subtomo0": torch.randn(2, N, N, N),
        "subtomo1": torch.randn(2, N, N, N),
        "ctf": torch.rand(2, N, N, N // 2 + 1).clamp(0, 1),
        "index": [0, 1],
    }
    loss, dc_loss, eq_loss = lit_unet._step(batch, batch_idx=0, deterministic=True)
    assert torch.allclose(loss, dc_loss + 2.0 * eq_loss)
    for term in (loss, dc_loss, eq_loss):
        assert torch.isfinite(term)
    # dc_loss's gradient must flow through x_hat_source
    assert dc_loss.requires_grad
