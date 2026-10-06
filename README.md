## DISCLAIMER

This project is forked from DeepDeWedge and changed to use 3D-CTFs as input. It still needs validation and updates of run instructions.

## Installation
We recommend installing into a fresh `Python >=3.10` environment, e.g. via [Anaconda](https://www.anaconda.com/download):
```
conda create -n helder -c conda-forge python=3.12 cuda-toolkit=12.9 -y
conda activate helder
```
Next, install torch-projectors which requires a specific version of pytorch:
```
python -m pip install torch==2.8.0
python -m pip install torch-projectors --index-url https://warpem.github.io/torch-projectors/cu129/simple/
```
Finally, install the Helder package directly from GitHub, which pulls in its remaining dependencies via `pyproject.toml`:
```
pip install git+https://github.com/McHaillet/helder
```
Upon successful installation, running
```
helder --help
```
should display a help message for the Helder command line interface.

## How-to

The repo is not fully cleaned up yet. To run the program:

1. run `scripts/make_even_odd_subtomos.py`
2. run `helder fit-model`
3. run `scripts/refine_tomogram_single.py` on all the tomograms you want to refine

Recommended is to fit a model with 64 channels and a subtomogram size of 96.

## License
All files are provided under the terms of the BSD 2-Clause license.
