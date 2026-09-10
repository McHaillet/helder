"""
End-to-end refinement of a single tilt series: reconstruct even/odd
subtomograms (+ shared 3D-CTF) on a regular grid of positions, refine each
pair with a fitted Helder U-Net, and reassemble the results into one
refined tomogram - all in memory, without ever writing individual subtomo
files to disk.

A grid of box centers is built that evenly covers the tilt series' volume
dimensions at `--box-size`, with boxes overlapping their neighbors by at
least `--overlap` (default 0.5) along each axis. At every grid position, an
even and an odd subtomogram are reconstructed at `--box-size` from the
movies' even/odd frame averages, plus a 3D-CTF (identical for even and odd,
since CTF does not depend on which half of the frames was used).

For each batch of grid positions, given the fitted model f, the even (y0)
and odd (y1) subtomograms are refined with one extra self-consistency pass
each - run through f, re-degraded with the reconstructed CTF, and run
through f again:

    x0 = f(apply_fourier_mask(f(y0), ctf))
    x1 = f(apply_fourier_mask(f(y1), ctf))
    refined = (x0 + x1) / 2

This mirrors the rotate + re-mask + refeed construction LitUnet3D._step uses
for its equivariance loss during fitting (minus the rotation), rather than
just running f once on each of y0/y1 and averaging.

Refined subtomograms are placed back into the output tomogram with
nearest-center (Voronoi) assignment by default - see
helder.utils.subtomos.reassemble_subtomos_nearest_center - not a blend of all
overlapping subtomos: subtomos are less reliable towards their own
edges/corners (e.g. correct_attenuation's sinc^2 correction grows with
distance from the reconstruction center), and blending several such
edge-degraded samples together compounds that degradation rather than
cancelling it. `--reassembly-method hann-ramp` opts into blending instead
(helder.utils.subtomos.reassemble_subtomos + get_hann_ramp_weights), which
can look smoother across seams at the cost of re-mixing in that edge
degradation. Since each subtomogram is reconstructed at a sub-voxel-precise
physical position (independent per-position backprojection, not cropped from
one pre-existing voxel grid), placement is necessarily an approximation: each
subtomogram is dropped into the output tomogram at its *nearest* integer
voxel corner. Fine for visually inspecting the result, not a sub-voxel-
accurate reconstruction.

Reconstruction and model inference both run on `--device` (default "cpu");
pass e.g. "cuda" or "cuda:0" to run on GPU. `--batch-size` caps how many grid
positions are reconstructed and refined in a single batch, bounding peak
device memory (default: no chunking, one batch for the whole tilt series).

`--oversampling` (default 3.0) is passed straight through to
reconstruct_subvolumes_single/reconstruct_subvolume_ctfs_single: it
backprojects from a `--box-size * --oversampling` patch and crops back to
`--box-size`, which is what gentles correct_attenuation's sinc^2 correction
near each box's own edges/corners - too low a value (e.g.
reconstruct_subvolumes_single's own default of 2.0) leaves every subtomo's
corners visibly boosted, which the nearest-center reassembly then avoids
compounding, but only where a nearer, less-boosted neighbor is available.

Usage:
    python refine_tomogram_single.py /path/to/tilt_series.xml --pixel-size 10.0 \\
        --box-size 96 --model-checkpoint logs/version_0/checkpoints/.../epoch=99.ckpt \\
        --output-file refined_tomogram.mrc --oversampling 3.0 \\
        --device cuda --batch-size 32
"""

import argparse
import math
from pathlib import Path

import torch
import tqdm

from warpylib import TiltSeries
from warpylib.ops import preprocess_tilt_data

from helder.fit_model import LitUnet3D
from helder.utils.fourier import apply_fourier_mask_to_tomo
from helder.utils.mrctools import save_mrc_data
from helder.utils.subtomos import reassemble_subtomos, reassemble_subtomos_nearest_center


def make_grid_axis_centers(volume_dims: torch.Tensor, box_physical: float, overlap: float) -> list:
    """
    Per-axis, evenly spaced box-center coordinates (Angstrom) that cover volume_dims,
    with neighboring boxes overlapping by at least `overlap`.
    """
    step = box_physical * (1.0 - overlap)
    axis_centers = []
    for length in volume_dims.tolist():
        if length <= box_physical:
            centers = torch.tensor([length / 2.0])
        else:
            n = math.ceil((length - box_physical) / step) + 1
            centers = torch.linspace(box_physical / 2.0, length - box_physical / 2.0, n)
        axis_centers.append(centers)
    return axis_centers


def axis_overlap_voxels(axis_centers: list, box_physical: float, pixel_size: float) -> list:
    """
    Actual overlap (in voxels) between neighboring grid boxes along each axis, derived from
    the real spacing between `axis_centers` rather than the nominal --overlap target: since
    the number of grid positions per axis is rounded up (see make_grid_axis_centers), the
    true spacing is often tighter than the nominal step, so the true overlap is >= the
    requested --overlap. get_hann_ramp_weights needs the true value - too narrow a ramp
    covers only part of the actual overlap and leaves a visible seam at every grid line.
    """
    overlaps = []
    for centers in axis_centers:
        if centers.numel() < 2:
            overlaps.append(0)
        else:
            spacing = (centers[1] - centers[0]).item()
            overlaps.append(max(0, round((box_physical - spacing) / pixel_size)))
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml_file", type=Path, help="Path to a single tilt series .xml file")
    parser.add_argument("--pixel-size", type=float, required=True, help="Reconstruction pixel size in Angstrom")
    parser.add_argument("--box-size", type=int, required=True, help="Subtomogram + CTF box size in pixels (must be even). Should match the subtomo_size used to fit --model-checkpoint")
    parser.add_argument("--model-checkpoint", type=Path, required=True, help="Path to a Helder model checkpoint (.ckpt)")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to save the refined tomogram (.mrc)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Minimum fractional overlap between neighboring grid positions, relative to --box-size (default: 0.5)")
    parser.add_argument("--reassembly-method", type=str, choices=["nearest-center", "hann-ramp"], default="nearest-center", help="How to combine overlapping refined subtomograms into the output tomogram. 'nearest-center' (default) assigns each voxel to its closest-center subtomogram - no blending, so it doesn't compound the edge/corner reconstruction degradation described above. 'hann-ramp' blends overlaps with Hann-shaped (raised-cosine) edge weights (helder.utils.subtomos.get_hann_ramp_weights), which can look smoother across seams but re-mixes in that edge degradation")
    parser.add_argument("--oversampling", type=float, default=3.0, help="Oversampling passed to reconstruct_subvolumes_single/reconstruct_subvolume_ctfs_single. Backprojects from a --box-size * --oversampling patch and crops back to --box-size, which gentles correct_attenuation's sinc^2 correction (it grows sharply towards each box's own corners) - too low a value leaves every subtomo's corners/edges visibly boosted (default: 3.0, vs. reconstruct_subvolumes_single's own default of 2.0)")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to reconstruct and run the model on, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Max grid positions reconstructed and refined in a single batch; splits large tomograms into chunks to bound memory use (default: no chunking)")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.box_size % 2 != 0:
        raise SystemExit("--box-size must be even")
    if not args.xml_file.is_file():
        raise SystemExit(f"No such file: {args.xml_file}")

    box_physical = args.box_size * args.pixel_size

    ts = TiltSeries(str(args.xml_file)).to(device)
    axis_centers = make_grid_axis_centers(ts.volume_dimensions_physical, box_physical, args.overlap)
    positions = torch.cartesian_prod(*axis_centers).reshape(-1, 3)
    print(f"{args.xml_file.name}: {positions.shape[0]} positions")

    # load_images always returns CPU tensors (mrcfile reads), so move them onto
    # `device` explicitly; ts itself already lives there via TiltSeries.to(device).
    positions_dev = positions.to(device)

    original_pixel_size = ts.ctf.pixel_size
    _, images_odd, images_even = ts.load_images(
        original_pixel_size=original_pixel_size,
        desired_pixel_size=args.pixel_size,
        load_averages=False,
        load_half_averages=True,
    )
    images_odd = images_odd.to(device)
    images_even = images_even.to(device)

    # Plane-subtraction + bandpass + normalization per tilt image, same preprocessing
    # warpylib's own reconstruct_full applies before backprojection.
    preprocess_size = args.box_size * args.oversampling
    images_odd = preprocess_tilt_data(images_odd, normalize=True, invert=False, subvolume_size=preprocess_size)
    images_even = preprocess_tilt_data(images_even, normalize=True, invert=False, subvolume_size=preprocess_size)

    model = LitUnet3D.load_from_checkpoint(args.model_checkpoint, map_location=device).to(device).eval()

    if args.batch_size is None:
        position_chunks = [positions_dev]
    else:
        position_chunks = [
            positions_dev[i:i + args.batch_size]
            for i in range(0, positions_dev.shape[0], args.batch_size)
        ]

    # Refined subtomograms, accumulated on CPU in grid order (position_chunks preserves
    # order, so a plain extend() keeps them index-matched to `positions`).
    refined_subtomos = []
    with torch.no_grad():
        for chunk in tqdm.tqdm(position_chunks, desc="Refining"):
            subtomo0 = ts.reconstruct_subvolumes_single(images_even, chunk, pixel_size=args.pixel_size, size=args.box_size, oversampling=args.oversampling, apply_ctf=True, correct_attenuation=True)
            subtomo1 = ts.reconstruct_subvolumes_single(images_odd, chunk, pixel_size=args.pixel_size, size=args.box_size, oversampling=args.oversampling, apply_ctf=True, correct_attenuation=True)
            ctf = ts.reconstruct_subvolume_ctfs_single(chunk, pixel_size=args.pixel_size, size=args.box_size, oversampling=args.oversampling, apply_ctf=True)

            x0 = model(subtomo0)
            x0 = apply_fourier_mask_to_tomo(x0, ctf)
            x0 = model(x0)

            x1 = model(subtomo1)
            x1 = apply_fourier_mask_to_tomo(x1, ctf)
            x1 = model(x1)

            refined_chunk = (x0 + x1) / 2
            refined_subtomos.extend(refined_chunk.cpu())

    # positions are box *centers*; convert to nearest-voxel start corners, clamped
    # to >= 0 for the edge case where the tilt series volume is smaller than one box
    start_coords = torch.round(positions / args.pixel_size - args.box_size / 2).clamp(min=0).to(torch.int64)
    tomo_shape = torch.round(ts.volume_dimensions_physical / args.pixel_size).to(torch.int64)
    # ts.volume_dimensions_physical (and therefore positions/start_coords/tomo_shape
    # derived from it above) is ordered X,Y,Z, but the subvolume tensors warpylib's
    # reconstruct_subvolumes_single actually returns are axis-ordered Z,Y,X.
    # reassemble_subtomos_nearest_center indexes tensors along their native axes, so
    # start_coords and tomo_shape must be reversed to Z,Y,X to line up with them.
    start_coords = start_coords.flip(-1)
    tomo_shape = tomo_shape.flip(-1)

    if args.reassembly_method == "nearest-center":
        tomo = reassemble_subtomos_nearest_center(
            subtomos=refined_subtomos,
            subtomo_start_coords=start_coords.tolist(),
            crop_to_size=tomo_shape.tolist(),
        )
    else:
        # axis_centers/overlap_voxels are ordered X,Y,Z like ts.volume_dimensions_physical;
        # flip to Z,Y,X to match the subtomo tensors' own axis order (see comment above).
        overlap_voxels = axis_overlap_voxels(axis_centers, box_physical, args.pixel_size)[::-1]
        tomo = reassemble_subtomos(
            subtomos=refined_subtomos,
            subtomo_start_coords=start_coords.tolist(),
            subtomo_overlap=overlap_voxels,
            crop_to_size=tomo_shape.tolist(),
        )

    print(f"Reassembled {len(refined_subtomos)} refined subvolume(s) into a tomogram of shape {tuple(tomo.shape)}.")
    print(f"Saving to '{args.output_file}'.")
    save_mrc_data(tomo.cpu(), str(args.output_file), save=True)


if __name__ == "__main__":
    main()
