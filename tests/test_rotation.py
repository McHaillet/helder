"""
Tests for helder.utils.rotation: the 20 grid-aligned rotations (get_grid_rotations,
sample_grid_rotation) and their exact, interpolation-free application (rotate_vol), and
rotation about the tilt axis by any angle (rotate_about_tilt_axis).
"""
import itertools
import math

import pytest
import torch

from helder.utils.rotation import (
    GRID_ROTATIONS,
    WEDGE_PRESERVING_FLIPS,
    get_grid_rotations,
    get_wedge_preserving_flips,
    rotate_about_tilt_axis,
    rotate_vol,
    sample_grid_rotation,
    sample_wedge_preserving_flip,
)


def _all_signed_permutation_rotations():
    """
    All 24 proper (det=+1) rotations that map a cubic voxel grid exactly onto itself.
    """
    mats = []
    for perm in itertools.permutations(range(3)):
        base = torch.eye(3)[list(perm)]
        for signs in itertools.product([1, -1], repeat=3):
            mat = base * torch.tensor(signs).reshape(3, 1)
            if round(torch.linalg.det(mat).item()) == 1:
                mats.append(mat.to(torch.int64))
    return mats


def _brute_force_rotate(vol, mat):
    """
    Reference implementation of rotate_vol via direct (slow) coordinate lookup, for a single
    3D volume - out[c] = vol[mat @ (c - center) + center].
    """
    n = vol.shape[0]
    center = (n - 1) / 2
    out = torch.zeros_like(vol)
    for i0 in range(n):
        for i1 in range(n):
            for i2 in range(n):
                coord = torch.tensor([i0, i1, i2], dtype=torch.float32) - center
                src = (mat.float() @ coord + center).round().long()
                out[i0, i1, i2] = vol[src[0], src[1], src[2]]
    return out


def test_get_grid_rotations_returns_20_of_the_24_proper_rotations():
    all_24 = _all_signed_permutation_rotations()
    grid_20 = get_grid_rotations()
    assert len(grid_20) == 20
    # all returned matrices are among the 24 grid-preserving proper rotations, with no dupes
    all_24_set = {tuple(m.flatten().tolist()) for m in all_24}
    grid_20_set = {tuple(m.flatten().tolist()) for m in grid_20}
    assert grid_20_set <= all_24_set
    assert len(grid_20_set) == 20


def test_get_grid_rotations_excludes_the_4_that_preserve_missing_wedge_direction():
    tilt_axis = 1
    excluded = set(_all_signed_permutation_rotations_as_tuples()) - {
        tuple(m.flatten().tolist()) for m in get_grid_rotations(tilt_axis=tilt_axis)
    }
    assert len(excluded) == 4
    perp = [d for d in range(3) if d != tilt_axis]
    for flat in excluded:
        mat = torch.tensor(flat).reshape(3, 3)
        # each excluded rotation fixes the tilt axis and doesn't swap the other two
        assert mat[tilt_axis, tilt_axis] != 0
        assert mat[perp[0], perp[1]] == 0


def _all_signed_permutation_rotations_as_tuples():
    return [tuple(m.flatten().tolist()) for m in _all_signed_permutation_rotations()]


def test_get_grid_rotations_result_independent_of_tilt_axis_choice():
    for tilt_axis in (0, 1, 2):
        assert len(get_grid_rotations(tilt_axis=tilt_axis)) == 20


def test_rotate_vol_matches_brute_force_for_all_20_rotations():
    torch.manual_seed(0)
    for n in (4, 5):  # even and odd sizes
        vol = torch.randn(n, n, n)
        for mat in GRID_ROTATIONS:
            expected = _brute_force_rotate(vol, mat)
            result = rotate_vol(vol, mat)
            assert result.shape == vol.shape
            assert torch.equal(result, expected)


def test_rotate_vol_preserves_leading_batch_dims():
    torch.manual_seed(0)
    vol = torch.randn(3, 2, 6, 6, 6)  # (batch, channel, Z, Y, X)
    for mat in GRID_ROTATIONS:
        result = rotate_vol(vol, mat)
        assert result.shape == vol.shape
        for b in range(vol.shape[0]):
            for c in range(vol.shape[1]):
                assert torch.equal(result[b, c], rotate_vol(vol[b, c], mat))


def test_rotate_vol_identity_is_a_no_op():
    torch.manual_seed(0)
    vol = torch.randn(6, 6, 6)
    identity = torch.eye(3, dtype=torch.int64)
    assert torch.equal(rotate_vol(vol, identity), vol)


def test_rotate_vol_round_trips_with_transpose():
    # for an orthogonal (signed permutation) matrix, the inverse is its transpose
    torch.manual_seed(0)
    vol = torch.randn(6, 6, 6)
    for mat in GRID_ROTATIONS:
        rotated = rotate_vol(vol, mat)
        back = rotate_vol(rotated, mat.T)
        assert torch.allclose(back, vol)


def test_rotate_vol_is_differentiable():
    vol = torch.randn(6, 6, 6, requires_grad=True)
    out = rotate_vol(vol, GRID_ROTATIONS[0])
    out.sum().backward()
    assert vol.grad is not None
    assert torch.allclose(vol.grad, torch.ones_like(vol))


def test_sample_grid_rotation_deterministic_is_reproducible():
    for index in range(5):
        a = sample_grid_rotation(index, deterministic=True)
        b = sample_grid_rotation(index, deterministic=True)
        assert torch.equal(a, b)


def test_sample_grid_rotation_returns_one_of_the_20():
    grid_set = {tuple(m.flatten().tolist()) for m in GRID_ROTATIONS}
    for index in range(20):
        mat = sample_grid_rotation(index, deterministic=True)
        assert tuple(mat.flatten().tolist()) in grid_set


def test_get_wedge_preserving_flips_are_the_3_missing_from_grid_rotations():
    """
    get_grid_rotations excludes exactly 4 of the 24 proper rotations (the ones that leave the
    missing-wedge/CTF direction unchanged); get_wedge_preserving_flips should return the
    non-identity 3 of those 4 (identity is a no-op, of no use as a "mirror" augmentation).
    """
    all_24 = {tuple(m.flatten().tolist()) for m in _all_signed_permutation_rotations()}
    grid_20 = {tuple(m.flatten().tolist()) for m in get_grid_rotations()}
    excluded_4 = all_24 - grid_20
    identity = tuple(torch.eye(3, dtype=torch.int64).flatten().tolist())
    assert excluded_4 - {identity} == {
        tuple(m.flatten().tolist()) for m in get_wedge_preserving_flips()
    }


def test_wedge_preserving_flips_are_diagonal_with_two_axes_flipped():
    for mat in WEDGE_PRESERVING_FLIPS:
        assert torch.equal(mat, torch.diag(torch.diagonal(mat)))  # diagonal, no permutation
        assert round(torch.linalg.det(mat.float()).item()) == 1
        assert (torch.diagonal(mat) == -1).sum().item() == 2


def test_wedge_preserving_flips_are_involutions():
    # each flips two axes together (each ±1 squared is 1), so applying one twice is identity
    for mat in WEDGE_PRESERVING_FLIPS:
        assert torch.equal(mat @ mat, torch.eye(3, dtype=torch.int64))


def test_rotate_vol_round_trips_for_wedge_preserving_flips():
    torch.manual_seed(0)
    vol = torch.randn(6, 6, 6)
    for mat in WEDGE_PRESERVING_FLIPS:
        flipped = rotate_vol(vol, mat)
        assert flipped.shape == vol.shape
        assert not torch.equal(flipped, vol)  # actually does something
        back = rotate_vol(flipped, mat.T)
        assert torch.allclose(back, vol)


def test_sample_wedge_preserving_flip_deterministic_is_reproducible():
    for index in range(5):
        a = sample_wedge_preserving_flip(index, deterministic=True)
        b = sample_wedge_preserving_flip(index, deterministic=True)
        assert torch.equal(a, b)


def test_sample_wedge_preserving_flip_returns_one_of_the_3():
    flip_set = {tuple(m.flatten().tolist()) for m in WEDGE_PRESERVING_FLIPS}
    for index in range(10):
        mat = sample_wedge_preserving_flip(index, deterministic=True)
        assert tuple(mat.flatten().tolist()) in flip_set


QUARTER_TURN = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]])


def _band_limited_blob(batch=1, size=32, seed=0):
    """
    Smooth random volumes confined to a central sphere, so that rotating them by shears is
    accurate (band-limited) and nothing wraps around the box.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch, size, size, size, generator=generator)
    coords = torch.arange(size) - (size - 1) / 2
    radius = (coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2).sqrt()
    sphere = torch.clamp((size / 4 - radius) / 2, 0, 1)
    # low-pass after masking, so that the result has (next to) nothing at Nyquist, which a
    # fractional shift can't represent exactly for an even size
    freqs = torch.fft.fftfreq(size)
    radius_ft = (freqs[:, None, None] ** 2 + freqs[None, :, None] ** 2 + freqs[None, None, :] ** 2).sqrt()
    lowpass = torch.exp(-((radius_ft / 0.12) ** 2))
    return torch.fft.ifftn(torch.fft.fftn(noise * sphere, dim=(-3, -2, -1)) * lowpass, dim=(-3, -2, -1)).real


def _rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("quarter_turns", [-2, -1, 0, 1, 2, 3, 4])
def test_rotate_about_tilt_axis_quarter_turns_match_grid_rotations(quarter_turns):
    vol = torch.randn(1, 12, 12, 12)
    rotated = rotate_about_tilt_axis(vol, torch.tensor([quarter_turns * math.pi / 2]))
    expected = vol[0]
    for _ in range(quarter_turns % 4):
        expected = rotate_vol(expected, QUARTER_TURN)
    assert torch.allclose(rotated[0], expected, atol=1e-5)


@pytest.mark.parametrize("deg_1, deg_2", [(30, 60), (45, 45), (170, 10), (-170, -10), (100, 80), (225, -45), (-100, 10)])
def test_rotate_about_tilt_axis_composes_to_exact_quarter_turns(deg_1, deg_2):
    # covers remainders on both sides of the nearest quarter turn, and angles beyond 180
    vol = _band_limited_blob()
    twice = rotate_about_tilt_axis(
        rotate_about_tilt_axis(vol, torch.tensor([math.radians(deg_1)])),
        torch.tensor([math.radians(deg_2)]),
    )
    expected = rotate_about_tilt_axis(vol, torch.tensor([math.radians(deg_1 + deg_2)]))
    assert _rel_err(twice, expected) < 1e-3


@pytest.mark.parametrize("deg", [10, 37, 45, 135, 170, -60])
def test_rotate_about_tilt_axis_round_trips(deg):
    vol = _band_limited_blob()
    theta = torch.tensor([math.radians(deg)])
    assert _rel_err(rotate_about_tilt_axis(rotate_about_tilt_axis(vol, theta), -theta), vol) < 1e-4


def test_rotate_about_tilt_axis_handles_mixed_angles_and_leading_dims():
    vol = _band_limited_blob(batch=6).reshape(2, 3, 32, 32, 32)
    theta = torch.tensor([[170.0, -100.0, 30.0], [260.0, -45.0, 90.0]]).deg2rad()
    rotated = rotate_about_tilt_axis(vol, theta)
    assert rotated.shape == vol.shape
    for i in range(2):
        for j in range(3):
            single = rotate_about_tilt_axis(vol[i, j][None], theta[i, j][None])[0]
            assert torch.allclose(rotated[i, j], single, atol=1e-6)


def test_rotate_about_tilt_axis_keeps_the_tilt_axis_fixed():
    # a volume that only varies along the tilt axis (Y) is unchanged by any rotation about it
    vol = torch.sin(torch.arange(16.0) / 3)[None, None, :, None].expand(1, 16, 16, 16)
    rotated = rotate_about_tilt_axis(vol, torch.tensor([math.radians(33)]))
    assert torch.allclose(rotated, vol, atol=1e-4)


def test_rotate_about_tilt_axis_is_differentiable():
    vol = _band_limited_blob().requires_grad_()
    rotate_about_tilt_axis(vol, torch.tensor([0.3])).sum().backward()
    assert vol.grad is not None and torch.isfinite(vol.grad).all()
