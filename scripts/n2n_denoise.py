"""
Plain Noise2Noise denoising of the even/odd subtomogram pairs written by
make_even_odd_subtomos.py - no CTF, no missing wedge, no equivariance term - as
a pre-processing step before fitting a model for CTF correction and missing
wedge filling.

Fitting: a helder.utils.unet.Unet3D f is fitted on the subtomo0/subtomo1 pairs
in `<subtomo-dir>/fitting_subtomos`, with the same random flip along each axis
applied to both halves of a pair, minimizing the symmetric N2N loss

    loss = MSE(f(subtomo0), subtomo1) + MSE(f(subtomo1), subtomo0)

MSE (not L1) so that f estimates the mean of the target given the input, i.e.
the clean signal. Only flips are used as augmentation: they keep the missing
wedge in place, whereas axis permutations would move it. After every epoch the
same loss is computed on `<subtomo-dir>/val_subtomos` (if present), and the
model with the lowest validation loss (or the last one, without validation
data) is saved to `<output-dir>/n2n_model.pt`. Pass `--model` to skip fitting
and only denoise with a previously saved model.

Denoising: every subtomo0/subtomo1 of both splits is denoised separately with
that model, and written to a new subtomo directory with the same layout, so it
can be passed straight to `helder fit-model --subtomo-dir`. The CTFs are copied
over unchanged:

    <output-dir>/{fitting,val}_subtomos/subtomo0/{0,1,...}.pt   (f(subtomo0))
    <output-dir>/{fitting,val}_subtomos/subtomo1/{0,1,...}.pt   (f(subtomo1))
    <output-dir>/{fitting,val}_subtomos/ctf/{0,1,...}.pt        (copied)

Usage:
    python n2n_denoise.py /path/to/subtomos --output-dir /path/to/subtomos_n2n \\
        --num-epochs 50 --batch-size 8 --device cuda
"""

import argparse
import random
import shutil
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from helder.utils.unet import Unet3D


class PairDataset(Dataset):
    """subtomo0/subtomo1 pairs of one split, optionally with the same random flips applied to both."""

    def __init__(self, split_dir: Path, flip: bool):
        self.split_dir = split_dir
        self.flip = flip
        self.n = len(list((split_dir / "subtomo0").glob("*.pt")))

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        subtomo0 = torch.load(self.split_dir / "subtomo0" / f"{index}.pt")
        subtomo1 = torch.load(self.split_dir / "subtomo1" / f"{index}.pt")
        if self.flip:
            flip_dims = [d for d in range(3) if random.random() < 0.5]
            subtomo0, subtomo1 = subtomo0.flip(flip_dims), subtomo1.flip(flip_dims)
        return subtomo0, subtomo1


def n2n_loss(model, subtomo0: torch.Tensor, subtomo1: torch.Tensor) -> torch.Tensor:
    inputs = torch.cat([subtomo0, subtomo1]).unsqueeze(1)
    targets = torch.cat([subtomo1, subtomo0]).unsqueeze(1)
    return 2 * torch.nn.functional.mse_loss(model(inputs), targets)


def fit(model, args, device, model_path: Path) -> None:
    def loader(split, train):
        return DataLoader(
            PairDataset(args.subtomo_dir / f"{split}_subtomos", flip=train),
            batch_size=args.batch_size, shuffle=train, drop_last=train,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True,
        )

    fitting_loader = loader("fitting", train=True)
    val_dir = args.subtomo_dir / "val_subtomos"
    val_loader = loader("val", train=False) if len(list((val_dir / "subtomo0").glob("*.pt"))) > 0 else None
    if val_loader is None:
        print("No validation data found, saving the model after the last epoch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        model.train()
        fitting_loss = 0.0
        for subtomo0, subtomo1 in tqdm.tqdm(fitting_loader, desc=f"Epoch {epoch}", leave=False):
            loss = n2n_loss(model, subtomo0.to(device), subtomo1.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            fitting_loss += loss.item() / len(fitting_loader)

        message = f"epoch {epoch}: fitting loss {fitting_loss:.5f}"
        checkpoint = {"unet_params": model_params(args), "state_dict": model.state_dict()}
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = sum(n2n_loss(model, s0.to(device), s1.to(device)).item() for s0, s1 in val_loader) / len(val_loader)
            message += f", val loss {val_loss:.5f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, model_path)
                message += " (saved)"
        elif epoch == args.num_epochs - 1:
            torch.save(checkpoint, model_path)
        print(message)


def model_params(args) -> dict:
    return {"chans": args.chans, "num_downsample_layers": args.num_downsample_layers}


@torch.no_grad()
def denoise_split(model, in_dir: Path, out_dir: Path, batch_size: int, device) -> None:
    """Denoise every subtomo0/subtomo1 in in_dir into out_dir, and copy the CTFs over."""
    n = len(list((in_dir / "subtomo0").glob("*.pt")))
    for name in ("subtomo0", "subtomo1"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        for start in tqdm.trange(0, n, batch_size, desc=f"Denoising {in_dir.name}/{name}"):
            indices = range(start, min(start + batch_size, n))
            batch = torch.stack([torch.load(in_dir / name / f"{i}.pt") for i in indices])
            denoised = model(batch.unsqueeze(1).to(device)).squeeze(1).cpu()
            for i, vol in zip(indices, denoised):
                torch.save(vol.clone(), out_dir / name / f"{i}.pt")
    shutil.copytree(in_dir / "ctf", out_dir / "ctf", dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subtomo_dir", type=Path, help="Subtomo directory written by make_even_odd_subtomos.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save the model and the new, denoised subtomo directory")
    parser.add_argument("--model", type=Path, default=None, help="Skip fitting and denoise with this previously saved model (n2n_model.pt)")
    parser.add_argument("--chans", type=int, default=32, help="Channels in the first U-Net layer (default: 32)")
    parser.add_argument("--num-downsample-layers", type=int, default=3, help="Number of U-Net downsampling layers; the subtomo size must be divisible by 2**this (default: 3)")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=8, help="Subtomo pairs per step; the model sees twice as many volumes, both directions (default: 8)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Adam learning rate (default: 4e-4)")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loading workers (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--device", type=str, default="cuda", help="torch device, e.g. 'cpu', 'cuda', 'cuda:0' (default: cuda)")
    args = parser.parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir.resolve() == args.subtomo_dir.resolve():
        raise SystemExit("--output-dir must differ from subtomo_dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "n2n_model.pt"
    if args.model is None:
        model = Unet3D(**model_params(args)).to(device)
        fit(model, args, device, model_path)
    else:
        model_path = args.model
    checkpoint = torch.load(model_path, map_location=device)
    model = Unet3D(**checkpoint["unet_params"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    for split in ("fitting", "val"):
        in_dir = args.subtomo_dir / f"{split}_subtomos"
        if in_dir.is_dir():
            denoise_split(model, in_dir, args.output_dir / f"{split}_subtomos", args.batch_size, device)


if __name__ == "__main__":
    main()
