"""
Tests for helder.utils.fourier: apply_fourier_mask_to_tomo (rfftn-based CTF/mask application)
and fft_3d (the fftshifted full transform, used only for visualization).
"""
import torch

from helder.utils.fourier import apply_fourier_mask_to_tomo, fft_3d


def test_apply_fourier_mask_matches_direct_rfftn_masking():
    torch.manual_seed(0)
    for N in [8, 12, 32]:
        mask = torch.rand(N, N, N // 2 + 1)
        vol = torch.randn(N, N, N)
        expected = torch.fft.irfftn(torch.fft.rfftn(vol, norm="ortho") * mask, s=(N, N, N), norm="ortho")
        result = apply_fourier_mask_to_tomo(vol, mask)
        assert torch.allclose(expected, result, atol=1e-5), N


def test_apply_fourier_mask_all_ones_is_identity():
    torch.manual_seed(0)
    N = 16
    vol = torch.randn(N, N, N)
    mask = torch.ones(N, N, N // 2 + 1)
    result = apply_fourier_mask_to_tomo(vol, mask)
    assert torch.allclose(vol, result, atol=1e-4)


def test_apply_fourier_mask_all_zeros_gives_zero():
    torch.manual_seed(0)
    N = 16
    vol = torch.randn(N, N, N)
    mask = torch.zeros(N, N, N // 2 + 1)
    result = apply_fourier_mask_to_tomo(vol, mask)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


def test_fft_3d_puts_dc_at_center_for_constant_input():
    N = 8
    vol = torch.ones(N, N, N)
    vol_ft = fft_3d(vol)
    dc = N // 2
    # a constant signal has all its energy at DC
    mask = torch.ones_like(vol_ft, dtype=torch.bool)
    mask[dc, dc, dc] = False
    assert vol_ft[dc, dc, dc].abs() > 0
    assert torch.allclose(vol_ft[mask], torch.zeros_like(vol_ft[mask]), atol=1e-6)
