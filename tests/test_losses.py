"""
Tests for helder.utils.losses: data_consistency_loss. Pure CPU/torch - no GPU needed.
"""
import torch

from helder.utils.fourier import apply_fourier_mask_to_tomo
from helder.utils.losses import data_consistency_loss


def test_data_consistency_loss_zero_for_perfect_cross_reconstruction():
    """
    If reconvolving x_hat with ctf exactly reproduces the (cross-wise) raw observation y,
    the loss must be exactly zero.
    """
    torch.manual_seed(0)
    N = 8
    ctf = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    x_hat = torch.randn(N, N, N)
    y = apply_fourier_mask_to_tomo(x_hat, ctf)
    loss = data_consistency_loss(x_hat, y, ctf)
    assert loss.item() < 1e-10


def test_data_consistency_loss_is_positive_otherwise():
    torch.manual_seed(0)
    N = 8
    ctf = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    x_hat = torch.randn(N, N, N)
    y = torch.randn(N, N, N)
    loss = data_consistency_loss(x_hat, y, ctf)
    assert loss.item() > 0


def test_data_consistency_loss_ignores_zero_ctf_frequencies():
    """
    Where ctf is exactly zero, the re-masked estimate is zero regardless of x_hat's value
    there, so the loss must not depend on x_hat at those frequencies as long as the raw
    observation agrees there too (also being subject to the same physical ctf). In rfftn
    convention every entry is independently maskable (no Hermitian-symmetry pairing to
    worry about, unlike the old full/fftshifted mask representation), so a perturbation with
    Fourier support exactly on ctf == 0 can be built directly.
    """
    torch.manual_seed(0)
    N = 8
    ctf = torch.zeros(N, N, N // 2 + 1)
    ctf[:, :, : N // 4] = 1.0

    x_hat_a = torch.randn(N, N, N)
    y = torch.zeros(N, N, N)

    delta_freq = torch.fft.rfftn(torch.randn(N, N, N), norm="ortho") * (1 - ctf)
    delta = torch.fft.irfftn(delta_freq, s=(N, N, N), norm="ortho")
    # sanity check: delta must indeed be invisible to ctf
    assert apply_fourier_mask_to_tomo(delta, ctf).abs().max().item() < 1e-4

    x_hat_b = x_hat_a + delta
    loss_a = data_consistency_loss(x_hat_a, y, ctf)
    loss_b = data_consistency_loss(x_hat_b, y, ctf)
    assert torch.allclose(loss_a, loss_b, atol=1e-4)


def test_data_consistency_loss_matches_manual_real_space_mae():
    torch.manual_seed(0)
    N = 8
    ctf = torch.rand(N, N, N // 2 + 1).clamp(0, 1)
    x_hat = torch.randn(N, N, N)
    y = torch.randn(N, N, N)
    expected = (apply_fourier_mask_to_tomo(x_hat, ctf) - y).abs().mean()
    assert torch.allclose(data_consistency_loss(x_hat, y, ctf), expected)
