import random

import pytorch_lightning as pl
import torch
from torch import nn

from .fourier import apply_fourier_mask_to_tomo, synthesize_noise
from .losses import data_consistency_loss
from .rotation import rotate_vol, sample_grid_rotation


class LitUnet3D(pl.LightningModule):
    """
    PyTrochLightning 'wrapper' of a 3D U-Net. This class implements steps for model fitting, validation and logging. This class is the heart of the 'helder fit-model' command.
    """

    def __init__(
        self,
        unet_params,
        adam_params,
        subtomo_size,
        lambda_=2.0,
    ):
        super().__init__()
        self.unet_params = unet_params
        self.adam_params = adam_params
        self.subtomo_size = subtomo_size
        self.lambda_ = lambda_
        self.unet = Unet3D(**self.unet_params)
        # self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x.unsqueeze(1)).squeeze(
            1
        )  # unsqueeze to add channel dimension, squeeze to remove it

    def _sample_rotations(self, indices, deterministic):
        """
        Samples one grid rotation per volume (see helder.utils.rotation.get_grid_rotations).
        Callers that need to later undo the same rotation (via _rotate_batch's 'inverse')
        must reuse the returned list rather than re-sampling by 'index': when 'deterministic'
        is False, sample_grid_rotation draws from the shared global 'random' state, so two
        separate calls - even with the same 'index' - are not guaranteed to agree.
        """
        return [sample_grid_rotation(int(index), deterministic) for index in indices]

    def _rotate_batch(self, vol_batch, rot_mats, inverse=False):
        """
        Rotates each volume in 'vol_batch' by the corresponding matrix in 'rot_mats' (from
        _sample_rotations). This is exact (no interpolation), so the output has exactly the
        same shape as the input and needs no cropping. If 'inverse' is True, applies the
        inverse rotation instead (the transpose of the signed permutation matrix, which
        exactly undoes rotate_vol for these grid-aligned rotations) - pass the *same*
        'rot_mats' used for the corresponding forward call so the two exactly cancel.
        """
        rotated = []
        for vol, rot_mat in zip(vol_batch, rot_mats):
            rotated.append(rotate_vol(vol, rot_mat.T if inverse else rot_mat))
        return torch.stack(rotated)

    def _step(self, batch, batch_idx, deterministic):
        subtomo0 = batch["subtomo0"]
        subtomo1 = batch["subtomo1"]
        ctf = batch["ctf"]

        # alternate which estimate is rotated + re-degraded to build the equivariance term's
        # model input ("x_hat2"). The same choice also picks which of the two cross-wise
        # data-consistency pairings this step evaluates (rather than always summing both):
        # both dc_loss and eq_loss compare against the same cross-wise raw measurement
        # 'y_cross' (see data_consistency_loss's docstring for why eq_loss is also just a
        # data-consistency comparison, on a different estimate of the same target).
        use_branch0_as_source = (
            (batch_idx % 2 == 0) if deterministic else (random.random() < 0.5)
        )
        x_hat_source = self(subtomo0 if use_branch0_as_source else subtomo1)
        y_cross = subtomo1 if use_branch0_as_source else subtomo0
        dc_loss = data_consistency_loss(x_hat_source, y_cross, ctf)

        # sampled once and reused for both the forward and inverse rotation below, so they
        # are guaranteed to exactly cancel (see _sample_rotations)
        rot_mats = self._sample_rotations(batch["index"], deterministic)

        # rotate_vol/_rotate_batch is differentiable, but x_hat_source is detached here
        # anyway by design (standard equivariant-imaging stop-gradient): the gradient of
        # eq_loss should only flow through the second application of self() below
        x_hat_source_rot = self._rotate_batch(x_hat_source.detach(), rot_mats)

        # the re-masked estimate is noise-free, unlike the raw observations the model sees in
        # its first application, so add synthetic noise to make it a new noisy observation.
        # The noise is added after rotating and re-masking, and is not rotated itself: it
        # belongs to the acquisition geometry, like 'ctf'
        generator = None
        if deterministic:
            generator = torch.Generator(device=subtomo0.device).manual_seed(
                int(batch["index"][0])
            )
        noise = synthesize_noise(subtomo0, subtomo1, generator=generator)
        z = apply_fourier_mask_to_tomo(x_hat_source_rot, ctf) + noise
        x_double_hat = self(z)
        x_double_hat_unrot = self._rotate_batch(x_double_hat, rot_mats, inverse=True)
        eq_loss = data_consistency_loss(x_double_hat_unrot, y_cross, ctf)

        loss = dc_loss + self.lambda_ * eq_loss
        return loss, dc_loss, eq_loss

    def training_step(self, batch, batch_idx):
        loss, dc_loss, eq_loss = self._step(batch, batch_idx, deterministic=False)
        # sync_dist=True: under multi-GPU DDP each rank only sees its own shard, so without
        # this the ModelCheckpoint callbacks that monitor "fitting_loss"/"val_loss" (see
        # fit_model.py) would select checkpoints based on a single rank's partial view of
        # the loss instead of the true value averaged across all ranks
        self.log("fitting_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("fitting_dc_loss", dc_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("fitting_eq_loss", eq_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dc_loss, eq_loss = self._step(batch, batch_idx, deterministic=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_dc_loss", dc_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_eq_loss", eq_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    # def on_before_zero_grad(self, optimizer) -> None:
    #     self.ema.update()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.adam_params)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return [optimizer]  # , [scheduler]

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
    #     if scheduler is not None:
    #         scheduler.step()


class Unet3D(torch.nn.Module):
    """
    PyTorch implementation of a 3D U-Net, which was inspired by the one used in the IsoNet software package (https://github.com/IsoNet-cryoET/IsoNet/tree/master/models/unet)
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_downsample_layers: int = 3,
        drop_prob: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_downsample_layers = num_downsample_layers
        self.drop_prob = drop_prob
        self.residual = residual
        self.__init_layers__()

    def __init_layers__(self):
        self.down_blocks = nn.ModuleList(
            [DownConvBlock(self.in_chans, self.chans, self.drop_prob)]
        )
        self.down_samplers = nn.ModuleList([SpatialDownSampling(self.chans)])

        ch = self.chans
        for _ in range(self.num_downsample_layers - 1):
            self.down_blocks.append(DownConvBlock(ch, ch * 2, self.drop_prob))
            self.down_samplers.append(SpatialDownSampling(ch * 2))
            ch *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch * 2, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(ch * 2, ch, kernel_size=(3, 3, 3), padding=1),
        )

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList([SpatialUpSampling(in_chans=ch, out_chans=ch)])

        for _ in range(self.num_downsample_layers - 1):
            self.up_blocks.append(UpConvBlock(2 * ch, ch, self.drop_prob))
            self.upsamplers.append(SpatialUpSampling(in_chans=ch, out_chans=ch // 2))
            ch //= 2
        self.up_blocks.append(UpConvBlock(2 * ch, ch, self.drop_prob))

        self.final_conv = nn.Conv3d(
            ch, self.out_chans, kernel_size=(1, 1, 1), stride=(1, 1, 1)
        )

    def get_loc_and_scale(self, volume: torch.Tensor):
        """
        Per-sample robust location and scale of 'volume', used to normalize each model input
        on its own rather than with dataset-wide statistics: the median, and half the
        distance between the 15.87th and 84.13th percentiles (equal to the standard deviation
        for Gaussian data, but insensitive to a few extreme voxels). Since forward() maps its
        output back with the same two values, the model is exactly equivariant to shifting
        and scaling its input, regardless of the biases and normalization layers inside.
        """
        flat = volume.flatten(1).sort(dim=1).values
        n = flat.shape[1]
        lo, loc, hi = (flat[:, round(q * (n - 1))] for q in (0.1587, 0.5, 0.8413))
        scale = ((hi - lo) / 2).clamp_min(1e-12)
        shape = (-1,) + (1,) * (volume.dim() - 1)
        return loc.view(shape), scale.view(shape)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        loc, scale = self.get_loc_and_scale(volume)
        volume = (volume - loc) / scale

        stack = []
        output = volume

        # apply down-sampling layers
        for block, downsampler in zip(self.down_blocks, self.down_samplers):
            output = block(output)
            stack.append(output)  # save intermediate outputs for skip connections
            output = downsampler(output)

        output = self.bottleneck(output)

        # apply up-sampling layers
        for upsampler, block in zip(self.upsamplers, self.up_blocks):
            output = upsampler(output, cat=stack.pop())
            output = block(output)

        output = self.final_conv(output)
        if self.residual:
            output = output + volume

        output = output * scale + loc
        return output


class DownConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        # the convs here have no bias of their own: the instance norm right after would
        # cancel it by subtracting the per-channel mean, so its affine shift is the bias
        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.InstanceNorm3d(out_chans, affine=True),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.InstanceNorm3d(out_chans, affine=True),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.InstanceNorm3d(out_chans, affine=True),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.layers(volume)


class UpConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, in_chans // 2, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.InstanceNorm3d(in_chans // 2, affine=True),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans // 2, in_chans // 2, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.InstanceNorm3d(in_chans // 2, affine=True),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans // 2, out_chans, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.InstanceNorm3d(out_chans, affine=True),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.layers(volume)


class SpatialDownSampling(nn.Module):
    def __init__(self, chans: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(chans, chans, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume):
        return self.layers(volume)


class SpatialUpSampling(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob=0.0):
        super().__init__()
        # Nearest-neighbor upsampling followed by a stride-1 conv, instead of a
        # strided ConvTranspose3d: with kernel_size=3 not divisible by stride=2, the
        # transposed conv gives uneven kernel overlap across output voxels, a
        # deterministic period-2 "checkerboard" artifact (see Odena et al., "Deconvolution
        # and Checkerboard Artifacts"). Resize+conv has no stride mismatch to produce
        # that unevenness.
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(
            in_chans, out_chans, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1
        )
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self, volume: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        output = self.conv(self.upsample(volume))
        output = torch.cat([output, cat], dim=1)
        output = self.activation(output)
        return output
