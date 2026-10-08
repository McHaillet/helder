"""
Tile pre-reconstructed even/odd tomograms into subtomograms and reconstruct a
matching 3D-CTF for every tile from the tilt series .xml.

Even and odd tomograms live in two separate directories with exactly matching
file names (e.g. tomo200528120_110_10.00Apx.mrc). Each pair is matched to its
tilt series .xml (e.g. tomo200528120_110.xml) by stripping the trailing
`_<pixel size>Apx` from the tomogram name.

Each tomogram is first normalized as a whole (mean subtracted, then divided
by its median absolute deviation scaled by 1.4826, a robust estimate of the
standard deviation), even and odd independently. Both are then tiled into `--box-size` cubes with the same 3D sliding
window helder's prepare-data uses (helder.utils.subtomos.extract_subtomos,
no padding), with stride `--stride` (default: `--box-size`, i.e. no overlap).
For every tile, its 3D center in Angstrom is computed from the tile's start
voxel and the tomogram's pixel size (read from the .mrc header), and a 3D-CTF
(identical for even and odd, since CTF does not depend on which half of the
frames was used) is reconstructed at that position at `--box-size` with
warpylib.

Per tomogram, tiles are randomly split into a fitting and a validation set
(`--val-fraction`, default 0.2). Output layout:

    <subtomo-dir>/fitting_subtomos/subtomo0/{0,1,...}.pt    (even, box-size)
    <subtomo-dir>/fitting_subtomos/subtomo1/{0,1,...}.pt    (odd, box-size)
    <subtomo-dir>/fitting_subtomos/ctf/{0,1,...}.pt         (box-size)
    <subtomo-dir>/val_subtomos/subtomo0/{0,1,...}.pt        (even, box-size)
    <subtomo-dir>/val_subtomos/subtomo1/{0,1,...}.pt        (odd, box-size)
    <subtomo-dir>/val_subtomos/ctf/{0,1,...}.pt             (box-size)

CTF reconstruction runs on `--device` (default "cpu"); pass e.g. "cuda" or
"cuda:0" to reconstruct on GPU. Saved .pt files are always moved back to CPU
first, so they load fine regardless of device. `--batch-size` caps how many
tile positions are reconstructed in a single call, chunking tomograms with
many tiles to bound peak device memory (default: no chunking, one call per
tomogram).

Usage:
    python make_even_odd_subtomos.py /path/to/even /path/to/odd /path/to/xmls \\
        --subtomo-dir /path/to/subtomos --box-size 96 --device cuda --batch-size 32
"""

import argparse
import math
import random
import re
from pathlib import Path

import mrcfile
import torch

from warpylib import TiltSeries

from helder.utils.mrctools import normalize_tomo
from helder.utils.subtomos import extract_subtomos


def load_tomo(mrc_file: Path) -> "tuple[torch.Tensor, float]":
    """Load a tomogram (axis order Z,Y,X) and its pixel size in Angstrom from the header."""
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        return torch.tensor(mrc.data).float(), float(mrc.voxel_size.x)


def tile_centers_physical(start_coords: list, box_size: int, pixel_size: float) -> torch.Tensor:
    """
    3D centers (Angstrom, ordered X,Y,Z) of the tiles extract_subtomos cut out at
    `start_coords`. extract_subtomos indexes the tomogram tensor along its native
    axes (Z,Y,X), while warpylib expects coordinates ordered X,Y,Z, hence the flip.
    """
    starts = torch.tensor(start_coords, dtype=torch.float32).reshape(-1, 3).flip(-1)
    return (starts + box_size / 2) * pixel_size


def batched_reconstruct(fn, positions: torch.Tensor, batch_size: "int | None") -> torch.Tensor:
    """
    Call fn(positions_chunk) over batch_size-sized chunks of positions (or all
    at once if batch_size is None), moving each chunk's result to CPU right
    away and concatenating into a single CPU tensor. Bounds peak device memory
    for tomograms with many tile positions.
    """
    if batch_size is None:
        return fn(positions).cpu()
    chunks = [fn(positions[i:i + batch_size]).cpu() for i in range(0, positions.shape[0], batch_size)]
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("even_dir", type=Path, help="Directory containing the even tomograms (.mrc)")
    parser.add_argument("odd_dir", type=Path, help="Directory containing the odd tomograms (.mrc), file names matching those in even_dir")
    parser.add_argument("xml_dir", type=Path, help="Directory containing the tilt series .xml files")
    parser.add_argument("--subtomo-dir", type=Path, required=True, help="Where to save the subtomograms and CTFs")
    parser.add_argument("--box-size", type=int, required=True, help="Subtomogram + CTF box size in pixels (must be even)")
    parser.add_argument("--stride", type=int, default=None, help="Stride in pixels of the sliding window used for tiling, same along all 3 axes (default: --box-size, i.e. no overlap)")
    parser.add_argument("--oversampling", type=float, default=2.0, help="Oversampling passed to reconstruct_subvolume_ctfs_single (default: 2.0)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of each tomogram's tiles assigned to the validation set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the fitting/validation split (default: 0)")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to reconstruct CTFs on, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Max tile positions reconstructed in a single CTF reconstruction call; splits large tomograms into chunks to bound memory use (default: no chunking)")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.box_size % 2 != 0:
        raise SystemExit("--box-size must be even")
    strides = None if args.stride is None else 3 * [args.stride]

    # Match every even tomogram to its odd tomogram (same file name) and its
    # tilt series .xml (same name minus the trailing "_<pixel size>Apx").
    even_paths = sorted(args.even_dir.glob("*.mrc"))
    if not even_paths:
        raise SystemExit(f"No .mrc files found in {args.even_dir}")
    entries = []  # (even_path, odd_path, xml_path)
    for even_path in even_paths:
        odd_path = args.odd_dir / even_path.name
        xml_path = args.xml_dir / (re.sub(r"_\d+(\.\d+)?Apx$", "", even_path.stem) + ".xml")
        for path in (odd_path, xml_path):
            if not path.exists():
                raise SystemExit(f"No match for {even_path}: {path} does not exist")
        entries.append((even_path, odd_path, xml_path))

    # Create the output directory layout.
    output_dirs = {}
    for split in ("fitting", "val"):
        split_dir = args.subtomo_dir / f"{split}_subtomos"
        output_dirs[split] = {name: split_dir / name for name in ("subtomo0", "subtomo1", "ctf")}
        for sub_dir in output_dirs[split].values():
            sub_dir.mkdir(parents=True, exist_ok=True)

    # Tile, reconstruct CTFs and save.
    rng = random.Random(args.seed)
    counters = {"fitting": 0, "val": 0}
    for entry_idx, (even_path, odd_path, xml_path) in enumerate(entries):
        even_tomo, pixel_size = load_tomo(even_path)
        odd_tomo, _ = load_tomo(odd_path)
        if even_tomo.shape != odd_tomo.shape:
            raise SystemExit(f"Shape mismatch between {even_path} {tuple(even_tomo.shape)} and {odd_path} {tuple(odd_tomo.shape)}")
        even_tomo = normalize_tomo(even_tomo)
        odd_tomo = normalize_tomo(odd_tomo)

        even_vols, start_coords = extract_subtomos(even_tomo, args.box_size, subtomo_extraction_strides=strides)
        odd_vols, _ = extract_subtomos(odd_tomo, args.box_size, subtomo_extraction_strides=strides)
        positions = tile_centers_physical(start_coords, args.box_size, pixel_size)
        print(f"[{entry_idx + 1}/{len(entries)}] {even_path.name}: {positions.shape[0]} tiles at {pixel_size} A/px")
        if positions.shape[0] == 0:
            continue

        # Shared 3D-CTF: identical for even and odd, so reconstructed once per position.
        ts = TiltSeries(str(xml_path)).to(device)
        # Tile positions are relative to the tomogram actually loaded, so use its
        # physical dimensions (X,Y,Z) rather than the ones recorded in the xml, same
        # as warpylib's own reconstruct_full does.
        ts.volume_dimensions_physical = torch.tensor(even_tomo.shape, dtype=torch.float32, device=device).flip(0) * pixel_size
        ctf_vols = batched_reconstruct(
            lambda p: ts.reconstruct_subvolume_ctfs_single(p, pixel_size=pixel_size, size=args.box_size, oversampling=args.oversampling, apply_ctf=True),
            positions.to(device), args.batch_size,
        )

        n_tiles = positions.shape[0]
        val_ids = set(rng.sample(range(n_tiles), math.ceil(n_tiles * args.val_fraction)))
        for local_idx in range(n_tiles):
            split = "val" if local_idx in val_ids else "fitting"
            dirs, out_idx = output_dirs[split], counters[split]
            torch.save(even_vols[local_idx].clone(), dirs["subtomo0"] / f"{out_idx}.pt")
            torch.save(odd_vols[local_idx].clone(), dirs["subtomo1"] / f"{out_idx}.pt")
            torch.save(ctf_vols[local_idx].clone(), dirs["ctf"] / f"{out_idx}.pt")
            counters[split] += 1

    print(f"Done: {counters['fitting']} fitting, {counters['val']} val subvolumes from {len(entries)} tomogram(s).")


if __name__ == "__main__":
    main()
