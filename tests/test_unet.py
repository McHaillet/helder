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


def test_forward_is_equivariant_to_input_shift_and_scale():
    """
    Each input is normalized with its own median/percentile scale and the output is mapped
    back with the same values (Unet3D.get_loc_and_scale), so shifting and scaling the input
    must shift and scale the output identically - despite the biases and instance norms
    inside the network.
    """
    torch.manual_seed(0)
    lit_unet = _make_lit_unet().eval()
    vol = torch.randn(2, 8, 8, 8)
    with torch.no_grad():
        out = lit_unet(vol)
        out_transformed = lit_unet(1e-4 * vol + 3.0)
    assert torch.allclose(out_transformed, 1e-4 * out + 3.0, atol=1e-6)


def test_loc_and_scale_ignore_outlier_voxels():
    torch.manual_seed(0)
    lit_unet = _make_lit_unet()
    vol = torch.randn(1, 1, 16, 16, 16)
    loc, scale = lit_unet.unet.get_loc_and_scale(vol)
    assert abs(loc.item()) < 0.1 and abs(scale.item() - 1.0) < 0.1
    vol_outliers = vol.clone()
    vol_outliers.view(-1)[:20] = 1000.0  # a few extreme voxels, e.g. a gold fiducial
    loc_outliers, scale_outliers = lit_unet.unet.get_loc_and_scale(vol_outliers)
    assert torch.allclose(loc_outliers, loc, atol=0.05)
    assert torch.allclose(scale_outliers, scale, atol=0.05)


def test_convs_followed_by_instance_norm_have_no_bias_and_all_others_do():
    lit_unet = _make_lit_unet()
    for block in [*lit_unet.unet.down_blocks, *lit_unet.unet.up_blocks]:
        layers = list(block.layers)
        for layer, next_layer in zip(layers, layers[1:]):
            if isinstance(layer, torch.nn.Conv3d):
                assert isinstance(next_layer, torch.nn.InstanceNorm3d)
                assert layer.bias is None and next_layer.bias is not None
    other_convs = [
        m
        for name, m in lit_unet.unet.named_modules()
        if isinstance(m, torch.nn.Conv3d) and "_blocks" not in name
    ]
    assert len(other_convs) > 0
    assert all(conv.bias is not None for conv in other_convs)


def test_step_is_reproducible_when_deterministic():
    """
    The synthetic noise added to the equivariance term's model input must not make the
    validation loss vary between calls.
    """
    torch.manual_seed(0)
    lit_unet = _make_lit_unet().eval()
    N = 8
    batch = {
        "subtomo0": torch.randn(2, N, N, N),
        "subtomo1": torch.randn(2, N, N, N),
        "ctf": torch.rand(2, N, N, N // 2 + 1).clamp(0, 1),
        "index": [0, 1],
    }
    with torch.no_grad():
        losses = [lit_unet._step(batch, batch_idx=0, deterministic=True)[0] for _ in range(2)]
    assert torch.equal(losses[0], losses[1])
