"""
Pull fitted Helder checkpoints from the cluster (if not already present locally),
run each on both halves (subtomo0/subtomo1) of the local subtomograms, average the two
outputs, and show every averaged subtomogram as a layer in a napari viewer, next to
the average of the two raw halves.

Also shown, to check the model input of the equivariance term during fitting (see
LitUnet3D._step): the prediction for subtomo0 with the 3D-CTF re-applied and synthetic
noise added ('simulated'), next to the raw subtomo0 ('half0') it should resemble. Unlike
during fitting, the prediction is not rotated first.

Usage (napari is not a helder dependency, so it is layered on top of the project venv):
    uv run --no-project --python .venv/bin/python --with "napari[pyqt6]" scripts/view_refined_subtomos.py
"""

import subprocess
from pathlib import Path

import napari
import torch

from helder.fit_model import LitUnet3D
from helder.utils.fourier import apply_fourier_mask_to_tomo, synthesize_noise

VERSION = "version_49"
REMOTE_CHECKPOINT_DIR = f"mchaillet@hpc04.tudelft.net:/home/mchaillet/data/match_pick_data/warp_tiltseries/logs/{VERSION}/checkpoints/val_loss"
CHECKPOINTS = [
    "epoch=39-val_loss=17.97027.ckpt",
]
SUBTOMO_DIR = Path("/home/marten/data/delft/ddw")


def load_models(device) -> dict:
    models = {}
    for checkpoint in CHECKPOINTS:
        local_checkpoint = SUBTOMO_DIR / f"{VERSION}_{checkpoint}"
        if not local_checkpoint.exists():
            subprocess.run(["scp", f"{REMOTE_CHECKPOINT_DIR}/{checkpoint}", str(local_checkpoint)], check=True)
        epoch = checkpoint.split("-")[0]
        models[epoch] = LitUnet3D.load_from_checkpoint(local_checkpoint, map_location=device).eval()
        print(f"{checkpoint}: lambda = {models[epoch].lambda_}")
    return models


def refine_subtomos() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    size = next(iter(models.values())).subtomo_size

    layers = {}
    files = sorted((SUBTOMO_DIR / "subtomo0").glob("*.pt"), key=lambda f: int(f.stem))
    with torch.no_grad():
        for file0 in files:
            if not (SUBTOMO_DIR / "ctf" / file0.name).exists():
                print(f"Skipping {file0.name}: no matching ctf")
                continue
            subtomo0 = torch.load(file0)
            subtomo1 = torch.load(SUBTOMO_DIR / "subtomo1" / file0.name)
            if subtomo0.shape != (size, size, size):
                print(f"Skipping {file0.name}: shape {tuple(subtomo0.shape)} does not match the model's subtomo_size {size}")
                continue
            ctf = torch.load(SUBTOMO_DIR / "ctf" / file0.name).to(device)
            ctf[0, 0, 0] = 1.0  # as in SubtomoDataset.__getitem__
            batch = torch.stack([subtomo0, subtomo1]).to(device)
            noise = synthesize_noise(batch[0], batch[1])
            layers[file0.stem] = {
                "raw": batch.mean(dim=0),
                # "half0": batch[0],
            }
            for epoch, model in models.items():
                prediction = model(batch)
                simulated = apply_fourier_mask_to_tomo(prediction[0], ctf, double_size=model.double_size_ctf) + noise
                layers[file0.stem][f"refined_{epoch}"] = prediction.mean(dim=0)
                # layers[file0.stem][f"simulated_{epoch}"] = simulated
    return layers


def main() -> None:
    viewer = napari.Viewer()
    for index, volumes in refine_subtomos().items():
        for name, volume in volumes.items():
            viewer.add_image(volume.cpu().numpy(), name=f"{name}_{index}")
    napari.run()


if __name__ == "__main__":
    main()
