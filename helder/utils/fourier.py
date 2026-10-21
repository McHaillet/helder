import functools

import torch
from torch import fft
from torch_grid_utils.shapes_3d import cuboid


def fft_3d(tomo, norm="ortho"):
    """
    3D Fourier transform with fftshift.
    """
    fft_dim = (-1, -2, -3)
    return fft.fftshift(fft.fftn(tomo, dim=fft_dim, norm=norm), dim=fft_dim)


@functools.lru_cache(maxsize=None)
def _padded_taper(size, device):
    """
    Soft mask for a (size, size, size) volume that was padded to twice its size: 1 on the
    volume itself, with a cosine falloff to 0 over the padding. Cached, since cuboid()
    computes the soft edge with a distance transform on the cpu.
    """
    # cuboid() excludes voxels exactly on its boundary, which for an even 'size' is the
    # first voxel of the volume along each axis, so make it one voxel larger to include it
    dimensions = (size + 1, size + 1, size + 1)
    # the first call may come from validation, which lightning runs in inference mode, and
    # a tensor created there could not be reused for model fitting afterwards
    with torch.inference_mode(False):
        return cuboid(
            dimensions=dimensions,
            image_shape=2 * size,
            smoothing_radius=size // 2,
            device=device,
        )


def apply_fourier_mask_to_tomo(tomo, mask, norm="ortho", double_size=False):
    """
    Multiplies the rfftn of 'tomo' with the real-valued CTF/mask 'mask' (rfftn convention:
    shape (..., N, N, N//2+1), unshifted, DC at index [..., 0, 0, 0]) and inverse-transforms
    back to real space. Used to apply/re-apply the CTF/missing-wedge mask. Operating
    directly in rfftn space (rather than expanding 'mask' to a full, fftshifted (N, N, N)
    array first) is both cheaper and simpler: 'mask' is never rotated in this codebase (only
    real-space volumes are), so there is no need for the fftshifted, DC-centered layout that
    rotation would require.

    If 'double_size' is True, 'mask' is instead sampled twice as finely in Fourier space as
    the transform of 'tomo' itself: for a 'tomo' of shape (..., N, N, N) it is in rfftn
    convention for a (2N, 2N, 2N) volume, i.e. shape (..., 2N, 2N, N+1). 'tomo' is then
    reflect-padded by N/2 on every side to (2N, 2N, 2N) and tapered to 0 over the padding
    (see _padded_taper) before 'mask' is applied, and the result is cropped back to
    (N, N, N). The finer sampling corresponds to a real-space kernel of twice the extent,
    and the padding gives that kernel room to spread signal beyond the box, rather than
    wrapping it around to the opposite side.
    """
    fft_dim = (-3, -2, -1)
    if not double_size:
        tomo_ft_masked = fft.rfftn(tomo, dim=fft_dim, norm=norm) * mask
        return fft.irfftn(tomo_ft_masked, s=tomo.shape[-3:], dim=fft_dim, norm=norm)

    size = tomo.shape[-1]
    pad = size // 2
    padded_shape = (2 * size,) * 3
    # reflect-pad by hand: the backward pass of torch's 3D reflection padding has no
    # deterministic cuda implementation, which fit_model's deterministic Trainer requires
    padded = tomo
    for dim in fft_dim:
        before = padded.narrow(dim, 1, pad).flip(dim)
        after = padded.narrow(dim, size - pad - 1, pad).flip(dim)
        padded = torch.cat([before, padded, after], dim=dim)
    padded = padded * _padded_taper(size, tomo.device)
    padded_ft_masked = fft.rfftn(padded, dim=fft_dim, norm=norm) * mask
    padded_masked = fft.irfftn(padded_ft_masked, s=padded_shape, dim=fft_dim, norm=norm)
    return padded_masked[..., pad:-pad, pad:-pad, pad:-pad]


def upsample_ctf(ctf, size):
    """
    Resamples the CTF/mask 'ctf' (rfftn convention, for a (N, N, N) volume) on the finer
    Fourier grid of a (size, size, size) volume, size > N, e.g. to get the twice finer
    sampling that apply_fourier_mask_to_tomo's 'double_size' needs from a CTF reconstructed
    at the box size. This zero-pads its point-spread function in real space, which is
    sinc interpolation: the values at the original sample points are kept exactly. The
    result is only as good as the PSF that 'ctf' holds: N samples can only represent a
    PSF of extent N, so tails beyond that are folded back into it rather than recovered.
    The sinc interpolation of sharp edges (e.g. of the missing wedge) ripples slightly, so
    the result is clamped to [0, 1], and the DC bin set to 1 as in SubtomoDataset.
    """
    n, dims = ctf.shape[-2], (-3, -2, -1)
    psf = fft.fftshift(fft.irfftn(ctf, s=(n,) * 3, dim=dims), dim=dims)  # lag 0 at index n//2
    if n % 2 == 0:
        # the Nyquist plane belongs to both +n/2 and -n/2: split it over both ends, so that
        # the PSF is symmetric about lag 0 once it doesn't wrap around anymore
        for dim in dims:
            psf = torch.cat([psf, psf.narrow(dim, 0, 1)], dim=dim)
            psf.narrow(dim, 0, 1).mul_(0.5)
            psf.narrow(dim, -1, 1).mul_(0.5)
    before = size // 2 - n // 2  # lag 0 at index size//2, where ifftshift expects it
    after = size - psf.shape[-1] - before
    padded = torch.nn.functional.pad(psf, (before, after) * 3)
    resampled = fft.rfftn(fft.ifftshift(padded, dim=dims), dim=dims).real.clamp(0, 1)
    resampled[..., 0, 0, 0] = 1.0
    return resampled


def synthesize_noise(subtomo0, subtomo1, generator=None):
    """
    Synthesizes a new noise realization from a pair of independent-noise observations of the
    same region. Their signal cancels in e = (subtomo1 - subtomo0) / sqrt(2), which leaves
    noise at the level of a single observation. The returned volume keeps the Fourier
    amplitudes of 'e' exactly and replaces its phases with random ones, so it has the same
    power in every frequency bin as the measured noise - including none where the missing
    wedge leaves none - while being uncorrelated with the noise in either observation.

    The random phases are taken from the transform of real white noise rather than drawn
    directly, which makes them Hermitian-symmetric by construction, so that the inverse
    transform is real without altering any amplitude. 'generator' makes the draw
    reproducible, e.g. for validation.
    """
    fft_dim = (-3, -2, -1)
    e_ft = fft.rfftn((subtomo1 - subtomo0) / 2**0.5, dim=fft_dim)
    white_noise = torch.randn(
        subtomo0.shape, generator=generator, device=subtomo0.device, dtype=subtomo0.dtype
    )
    w_ft = fft.rfftn(white_noise, dim=fft_dim)
    phase = w_ft / w_ft.abs().clamp_min(1e-30)
    return fft.irfftn(e_ft.abs() * phase, s=subtomo0.shape[-3:], dim=fft_dim)
