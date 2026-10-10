import torch

from .fourier import apply_fourier_mask_to_tomo


def data_consistency_loss(x_hat, y, ctf):
    """
    Noise2Noise-style data-consistency loss, for one estimate/observation pair. 'x_hat' is
    the model's estimate from one of the two independent-noise raw observations; 'y' is the
    *other* one (which already carries the same physical 'ctf' baked in from acquisition/
    reconstruction - it is never re-applied to it). 'x_hat' is re-masked with the canonical
    (native-orientation) 'ctf' and compared against 'y' - the caller is responsible for this
    cross-wise pairing (passing the observation 'x_hat' did *not* come from), since comparing
    an estimate to the observation it came from would let the model trivially learn identity
    without ever averaging out noise.

    Because both raw observations share the exact same physical 'ctf', frequencies where
    'ctf' is near zero contribute ~0 automatically (both the re-masked estimate and the raw
    target are ~0 there) - unlike the old two-region masked_loss, no extra region weighting
    is needed to handle this, which matters for a continuous (non-binary) CTF.

    LitUnet3D._step uses this for its "dc_loss" only: the equivariance term ("eq_loss") is a
    plain L2 loss between two estimates of the model, with no 'ctf' involved.
    """
    residual = apply_fourier_mask_to_tomo(x_hat, ctf) - y
    return residual.square().mean()
