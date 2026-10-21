import itertools
import math
import random

import torch

# Sub-tomograms are stored axis-ordered (Z, Y, X) throughout this codebase (see e.g.
# scripts/refine_tomogram_single.py), and the tilt series is acquired by tilting around the Y
# axis - dim 1 of the last 3 dims.
TILT_AXIS = 1

BASE_SEED = 888


def get_grid_rotations(tilt_axis=TILT_AXIS):
    """
    Returns the 20 rotations that can be applied to a cubic sub-tomogram exactly, without any
    interpolation, and that actually change its missing-wedge/CTF direction relative to
    'tilt_axis'. Each rotation is a signed permutation matrix (a permutation of the 3 axes
    with independent +-1 sign flips) - the full set of 24 such matrices are exactly the
    proper (det=+1) rotations that map a cubic voxel grid exactly onto itself. Of those 24, 4
    only rotate the volume around 'tilt_axis' (by 0 or 180 degrees) or flip it along
    'tilt_axis' without otherwise touching the missing-wedge/CTF's orientation, so they leave
    it unchanged; the remaining 20 are returned here. This follows IsoNet (Supplementary Fig.
    2), which uses the same 20 grid-aligned rotations for interpolation-free data
    augmentation.
    """
    perp = [d for d in range(3) if d != tilt_axis]
    rotations = []
    for perm in itertools.permutations(range(3)):
        base = torch.eye(3)[list(perm)]
        for signs in itertools.product([1, -1], repeat=3):
            mat = (base * torch.tensor(signs).reshape(3, 1)).to(torch.int64)
            if round(torch.linalg.det(mat.float()).item()) != 1:
                continue  # keep only proper rotations
            tilt_axis_fixed = mat[tilt_axis, tilt_axis] != 0
            perp_axes_swapped = mat[perp[0], perp[1]] != 0
            if tilt_axis_fixed and not perp_axes_swapped:
                continue  # leaves the missing-wedge/CTF direction unchanged
            rotations.append(mat)
    return rotations


GRID_ROTATIONS = get_grid_rotations()


def sample_grid_rotation(index, deterministic):
    """
    Samples one of the 20 grid rotations (see get_grid_rotations). If 'deterministic' is
    True, the choice only depends on 'index' (via BASE_SEED + index), giving reproducible
    rotations, e.g. for validation.
    """
    rng = random.Random(BASE_SEED + index) if deterministic else random
    return rng.choice(GRID_ROTATIONS)


def get_wedge_preserving_flips():
    """
    Returns the complement of get_grid_rotations' 20 rotations: the 3 non-identity diagonal
    sign matrices with det=+1 (each flips exactly two of the three axes, keeping the third
    fixed). A diagonal matrix never swaps axes, so the tilt axis always maps to itself
    (at most sign-flipped) and the two perpendicular axes never trade places - per
    get_grid_rotations' docstring, this combination leaves the missing-wedge/CTF orientation
    unchanged (a symmetric +-tilt-angle wedge is invariant under negating the tilt axis, and
    mirroring the two perpendicular axes together doesn't touch it either), regardless of
    which axis is 'tilt_axis'.

    Exact and interpolation-free (via rotate_vol) like get_grid_rotations, but for the
    opposite reason they're useful: since they don't change the wedge orientation, they're
    useless for LitUnet3D._step's equivariance term, but are exactly what a "mirror" data
    augmentation needs -
    they still displace any fixed, position-anchored network artifact (e.g. conv boundary or
    upsampling-checkerboard bias) without invalidating the shared 'ctf' tensor or pushing the
    model to a wedge orientation it never saw in training.
    """
    return [
        torch.diag(torch.tensor(signs, dtype=torch.int64))
        for signs in itertools.product([1, -1], repeat=3)
        if signs.count(-1) == 2
    ]


WEDGE_PRESERVING_FLIPS = get_wedge_preserving_flips()

# separate seed base from BASE_SEED so sampling a flip and a rotation for the same 'index'
# don't draw from identically-seeded RNG states
FLIP_BASE_SEED = 999


def sample_wedge_preserving_flip(index, deterministic):
    """
    Samples one of the 3 wedge-preserving flips (see get_wedge_preserving_flips), the same way
    sample_grid_rotation samples from GRID_ROTATIONS.
    """
    rng = random.Random(FLIP_BASE_SEED + index) if deterministic else random
    return rng.choice(WEDGE_PRESERVING_FLIPS)


def rotate_vol(vol, rot_mat):
    """
    Applies the grid rotation 'rot_mat' (one of get_grid_rotations()'s 20 signed permutation
    matrices) to the last 3 dims of 'vol' by permuting/flipping those axes. Since 'rot_mat'
    maps the voxel grid exactly onto itself, this is exact (no interpolation, unlike the old
    scipy.ndimage.affine_transform-based approach) and the output has exactly the same shape
    as the input - no cropping needed. Unlike that old approach, this is fully differentiable
    and needs no CPU round-trip.
    """
    lead = vol.dim() - 3
    # perm[i]: which of the 3 trailing input axes feeds output axis i
    perm = rot_mat.abs().argmax(dim=1).tolist()
    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[p] = i
    flip_dims = [lead + i for i in range(3) if rot_mat[i, perm[i]] < 0]
    if flip_dims:
        vol = vol.flip(dims=flip_dims)
    return vol.permute(*range(lead), *(lead + p for p in inv_perm))


def _shear(vol, shear, along, by):
    """
    Shears the (B, n0, n1, n2) volumes 'vol' along dim 'along' (1, 2 or 3) in proportion to
    the coordinate along dim 'by': each line along 'along' is shifted by
    shear[b] * (coordinate - center), where 'shear' is (B,) and the center is that of the
    voxel grid. The shift is a phase ramp on a 1D rfft along 'along', which is exact (sinc
    interpolation) for band-limited content, but wraps around the box. For an even size the
    Nyquist bin can't hold a fractional shift (irfft drops its imaginary part), so content
    at Nyquist is slightly damped.
    """
    n_along, n_by = vol.shape[along], vol.shape[by]
    freqs = torch.fft.rfftfreq(n_along, device=vol.device)  # cycles / voxel
    coords = torch.arange(n_by, device=vol.device, dtype=vol.dtype) - (n_by - 1) / 2
    angles = -2 * math.pi * shear[:, None, None] * coords[None, :, None] * freqs  # (B, n_by, K)
    # place the n_by and K axes of the phase ramp at dims 'by' and 'along' of 'vol'
    shape = [len(vol), 1, 1, 1]
    shape[by], shape[along] = n_by, len(freqs)
    phase = torch.polar(torch.ones_like(angles), angles)
    if by > along:
        phase = phase.transpose(1, 2)
    ft = torch.fft.rfft(vol, dim=along) * phase.reshape(shape)
    return torch.fft.irfft(ft, n=n_along, dim=along)


def rotate_about_tilt_axis(vol, theta, tilt_axis=TILT_AXIS):
    """
    Rotates the last 3 dims of 'vol' about 'tilt_axis' by any angle 'theta' (in radians, a
    tensor of shape vol.shape[:-3], one angle per volume). A positive quarter turn equals the
    grid rotation [[0, 0, -1], [0, 1, 0], [1, 0, 0]] for the default 'tilt_axis' (via
    rotate_vol).

    Each angle is split into the nearest multiple of 90 degrees, applied exactly with
    torch.rot90, and a remainder of at most 45 degrees, applied as three shears (Paeth's
    decomposition), each an exact Fourier phase shift (see _shear). Unlike trilinear
    interpolation this keeps all frequency amplitudes, up to Nyquist, but the shears wrap
    content around the box: only the central region of the output is valid, so pad the
    volume (e.g. extract it at ~sqrt(2) times the size needed) and crop after rotating.
    Differentiable.
    """
    lead_shape = vol.shape[:-3]
    vol = vol.reshape(-1, *vol.shape[-3:])
    theta = theta.reshape(-1).to(device=vol.device, dtype=vol.dtype)
    # dims (of the batched 4D volume) of the rotation plane, in the order that matches the
    # grid rotation documented above
    plane_0, plane_1 = (1 + d for d in range(3) if d != tilt_axis)

    quarter_turns = torch.round(theta / (math.pi / 2))
    rest = theta - quarter_turns * (math.pi / 2)
    a, b = -torch.tan(rest / 2), torch.sin(rest)
    vol = _shear(vol, a, along=plane_1, by=plane_0)
    vol = _shear(vol, b, along=plane_0, by=plane_1)
    vol = _shear(vol, a, along=plane_1, by=plane_0)

    # torch.rot90 takes a python int, so rotate the volumes with the same number of quarter
    # turns together
    rotated = []
    order = []
    for k in quarter_turns.unique().tolist():
        idx = (quarter_turns == k).nonzero().squeeze(1)
        rotated.append(torch.rot90(vol[idx], int(k) % 4, dims=(plane_1, plane_0)))
        order.append(idx)
    vol = torch.cat(rotated)[torch.argsort(torch.cat(order))]
    return vol.reshape(*lead_shape, *vol.shape[-3:])
