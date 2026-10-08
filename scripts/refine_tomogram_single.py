"""
End-to-end refinement of a single tomogram from its even/odd pair: tile the
two pre-reconstructed half tomograms into subtomograms, refine each pair with
a fitted Helder U-Net, and reassemble the results into one refined tomogram -
all in memory, without ever writing individual subtomo files to disk.

Each tomogram is first normalized as a whole exactly as
make_even_odd_subtomos.py does before tiling the fitting data (mean
subtracted, then divided by its median absolute deviation scaled by 1.4826),
even and odd independently, so the model sees inputs on the scale it was
fitted on. The refined tomogram is saved on that normalized scale.

A grid of `--box-size` tiles is built that evenly covers the whole tomogram,
with tiles overlapping their neighbors by at least `--overlap` (default 0.5)
along each axis. Given the fitted model f, the even (y0) and odd (y1) tile
at every grid position are refined and averaged:

    refined = (f(y0) + f(y1)) / 2

Refined subtomograms are placed back into the output tomogram with
nearest-center (Voronoi) assignment by default - see
helder.utils.subtomos.reassemble_subtomos_nearest_center - not a blend of all
overlapping subtomos: the model's output is less reliable towards each
subtomo's own edges/corners, and blending several such edge-degraded samples
together compounds that degradation rather than cancelling it.
`--reassembly-method hann-ramp` opts into blending instead
(helder.utils.subtomos.reassemble_subtomos + get_hann_ramp_weights), which
can look smoother across seams at the cost of re-mixing in that edge
degradation.

Model inference runs on `--device` (default "cpu"); pass e.g. "cuda" or
"cuda:0" to run on GPU. `--batch-size` caps how many tiles are refined in a
single batch, bounding peak device memory (default: no chunking, one batch
for the whole tomogram).

Usage:
    python refine_tomogram_single.py /path/to/even.mrc /path/to/odd.mrc \\
        --box-size 96 --model-checkpoint logs/version_0/checkpoints/.../epoch=99.ckpt \\
        --output-file refined_tomogram.mrc --device cuda --batch-size 32
"""

import argparse
import math
from pathlib import Path

import torch
import tqdm

from helder.fit_model import LitUnet3D
from helder.utils.mrctools import load_mrc_data, normalize_tomo, save_mrc_data
from helder.utils.subtomos import reassemble_subtomos, reassemble_subtomos_nearest_center


def make_grid_axis_starts(tomo_shape: tuple, box_size: int, overlap: float) -> list:
    """
    Per-axis, evenly spaced tile start voxels that cover tomo_shape, with neighboring
    tiles overlapping by at least `overlap` (as a fraction of box_size).
    """
    step = box_size * (1.0 - overlap)
    axis_starts = []
    for length in tomo_shape:
        n = math.ceil((length - box_size) / step) + 1
        axis_starts.append(torch.linspace(0, length - box_size, n).round().to(torch.int64))
    return axis_starts


def axis_overlap_voxels(axis_starts: list, box_size: int) -> list:
    """
    Actual overlap (in voxels) between neighboring tiles along each axis, derived from
    the real spacing between `axis_starts` rather than the nominal --overlap target: since
    the number of grid positions per axis is rounded up (see make_grid_axis_starts), the
    true spacing is often tighter than the nominal step, so the true overlap is >= the
    requested --overlap. get_hann_ramp_weights needs the true value - too narrow a ramp
    covers only part of the actual overlap and leaves a visible seam at every grid line.
    """
    overlaps = []
    for starts in axis_starts:
        if starts.numel() < 2:
            overlaps.append(0)
        else:
            overlaps.append(max(0, box_size - (starts[1] - starts[0]).item()))
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("even_tomo", type=Path, help="Path to the even tomogram (.mrc)")
    parser.add_argument("odd_tomo", type=Path, help="Path to the odd tomogram (.mrc)")
    parser.add_argument("--box-size", type=int, required=True, help="Subtomogram box size in pixels (must be even). Should match the subtomo_size used to fit --model-checkpoint")
    parser.add_argument("--model-checkpoint", type=Path, required=True, help="Path to a Helder model checkpoint (.ckpt)")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to save the refined tomogram (.mrc)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Minimum fractional overlap between neighboring grid positions, relative to --box-size (default: 0.5)")
    parser.add_argument("--reassembly-method", type=str, choices=["nearest-center", "hann-ramp"], default="nearest-center", help="How to combine overlapping refined subtomograms into the output tomogram. 'nearest-center' (default) assigns each voxel to its closest-center subtomogram - no blending, so it doesn't compound the edge/corner degradation described above. 'hann-ramp' blends overlaps with Hann-shaped (raised-cosine) edge weights (helder.utils.subtomos.get_hann_ramp_weights), which can look smoother across seams but re-mixes in that edge degradation")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to run the model on, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Max tiles refined in a single batch; splits large tomograms into chunks to bound memory use (default: no chunking)")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.box_size % 2 != 0:
        raise SystemExit("--box-size must be even")

    even_tomo = load_mrc_data(args.even_tomo).float()
    odd_tomo = load_mrc_data(args.odd_tomo).float()
    if even_tomo.shape != odd_tomo.shape:
        raise SystemExit(f"Shape mismatch between {args.even_tomo} {tuple(even_tomo.shape)} and {args.odd_tomo} {tuple(odd_tomo.shape)}")
    tomo_shape = tuple(even_tomo.shape)
    if min(tomo_shape) < args.box_size:
        raise SystemExit(f"--box-size {args.box_size} is larger than the tomogram {tomo_shape}")
    even_tomo = normalize_tomo(even_tomo)
    odd_tomo = normalize_tomo(odd_tomo)

    # Tile start voxels, along the tomogram tensor's native axes (Z,Y,X).
    axis_starts = make_grid_axis_starts(tomo_shape, args.box_size, args.overlap)
    start_coords = torch.cartesian_prod(*axis_starts).reshape(-1, 3).tolist()
    print(f"{args.even_tomo.name}: {len(start_coords)} positions")

    model = LitUnet3D.load_from_checkpoint(args.model_checkpoint, map_location=device).to(device).eval()

    batch_size = len(start_coords) if args.batch_size is None else args.batch_size
    start_chunks = [start_coords[i:i + batch_size] for i in range(0, len(start_coords), batch_size)]

    def extract(tomo: torch.Tensor, starts: list) -> torch.Tensor:
        b = args.box_size
        return torch.stack([tomo[z:z + b, y:y + b, x:x + b] for z, y, x in starts]).to(device)

    # Refined subtomograms, accumulated on CPU in grid order (start_chunks preserves
    # order, so a plain extend() keeps them index-matched to `start_coords`).
    refined_subtomos = []
    with torch.no_grad():
        for chunk in tqdm.tqdm(start_chunks, desc="Refining"):
            x0 = model(extract(even_tomo, chunk))
            x1 = model(extract(odd_tomo, chunk))

            refined_chunk = (x0 + x1) / 2
            refined_subtomos.extend(refined_chunk.cpu())

    if args.reassembly_method == "nearest-center":
        tomo = reassemble_subtomos_nearest_center(
            subtomos=refined_subtomos,
            subtomo_start_coords=start_coords,
            crop_to_size=tomo_shape,
        )
    else:
        tomo = reassemble_subtomos(
            subtomos=refined_subtomos,
            subtomo_start_coords=start_coords,
            subtomo_overlap=axis_overlap_voxels(axis_starts, args.box_size),
            crop_to_size=tomo_shape,
        )

    print(f"Reassembled {len(refined_subtomos)} refined subvolume(s) into a tomogram of shape {tuple(tomo.shape)}.")
    print(f"Saving to '{args.output_file}'.")
    save_mrc_data(tomo.cpu(), str(args.output_file), save=True)


if __name__ == "__main__":
    main()
