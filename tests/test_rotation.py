"""
Tests for helder.utils.rotation: the 20 grid-aligned rotations (get_grid_rotations,
sample_grid_rotation) and their exact, interpolation-free application (rotate_vol).
"""
import itertools

import torch

from helder.utils.rotation import (
    GRID_ROTATIONS,
    WEDGE_PRESERVING_FLIPS,
    get_grid_rotations,
    get_wedge_preserving_flips,
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
