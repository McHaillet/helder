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
