"""
Tests for helder.utils.fourier: apply_fourier_mask_to_tomo (rfftn-based CTF/mask application)
fft_3d (the fftshifted full transform, used only for visualization) and synthesize_noise
(phase-randomized noise from a pair of observations).
"""
import torch

from helder.utils.fourier import apply_fourier_mask_to_tomo, fft_3d, synthesize_noise


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


def _noisy_pair(N, batch=2):
    signal = torch.randn(batch, N, N, N)
    return signal + torch.randn(batch, N, N, N), signal + torch.randn(batch, N, N, N)


def test_synthesize_noise_keeps_amplitude_spectrum_of_half_difference():
    torch.manual_seed(0)
    for N in [8, 12, 32]:
        y0, y1 = _noisy_pair(N)
        e = (y1 - y0) / 2**0.5
        noise = synthesize_noise(y0, y1)
        assert noise.shape == y0.shape and not torch.is_complex(noise)
        e_amp = torch.fft.rfftn(e, dim=(-3, -2, -1)).abs()
        noise_amp = torch.fft.rfftn(noise, dim=(-3, -2, -1)).abs()
        assert torch.allclose(noise_amp, e_amp, rtol=1e-3, atol=1e-3 * e_amp.max()), N


def test_synthesize_noise_is_zero_where_the_half_difference_has_no_power():
    # mimics a missing wedge: both observations lack the same frequency bins
    torch.manual_seed(0)
    N = 16
    mask = torch.ones(N, N, N // 2 + 1)
    mask[..., 2:6] = 0
    y0, y1 = (apply_fourier_mask_to_tomo(y, mask) for y in _noisy_pair(N))
    noise_amp = torch.fft.rfftn(synthesize_noise(y0, y1), dim=(-3, -2, -1)).abs()
    assert noise_amp[:, mask == 0].max() < 1e-3 * noise_amp.max()


def test_synthesize_noise_is_uncorrelated_with_inputs_and_between_draws():
    torch.manual_seed(0)
    y0, y1 = _noisy_pair(32, batch=1)
    e = (y1 - y0) / 2**0.5
    noise, noise2 = synthesize_noise(y0, y1), synthesize_noise(y0, y1)
    corr = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].abs()
    for other in (e, y0, y1, noise2):
        assert corr(noise, other) < 0.05


def test_synthesize_noise_is_reproducible_with_generator():
    torch.manual_seed(0)
    y0, y1 = _noisy_pair(8)
    draws = [synthesize_noise(y0, y1, generator=torch.Generator().manual_seed(3)) for _ in range(2)]
    assert torch.equal(draws[0], draws[1])
