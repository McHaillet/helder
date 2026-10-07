import torch
from torch import fft


def fft_3d(tomo, norm="ortho"):
    """
    3D Fourier transform with fftshift.
    """
    fft_dim = (-1, -2, -3)
    return fft.fftshift(fft.fftn(tomo, dim=fft_dim, norm=norm), dim=fft_dim)


def apply_fourier_mask_to_tomo(tomo, mask, norm="ortho"):
    """
    Multiplies the rfftn of 'tomo' with the real-valued CTF/mask 'mask' (rfftn convention:
    shape (..., N, N, N//2+1), unshifted, DC at index [..., 0, 0, 0]) and inverse-transforms
    back to real space. Used to apply/re-apply the CTF/missing-wedge mask. Operating
    directly in rfftn space (rather than expanding 'mask' to a full, fftshifted (N, N, N)
    array first) is both cheaper and simpler: 'mask' is never rotated in this codebase (only
    real-space volumes are), so there is no need for the fftshifted, DC-centered layout that
    rotation would require.
    """
    fft_dim = (-3, -2, -1)
    tomo_ft_masked = fft.rfftn(tomo, dim=fft_dim, norm=norm) * mask
    return fft.irfftn(tomo_ft_masked, s=tomo.shape[-3:], dim=fft_dim, norm=norm)


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
