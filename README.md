## DISCLAIMER

This project is forked from DeepDeWedge and changed to use 3D-CTFs as input. It still needs validation and updates of run instructions.

## Installation
We recommend installing into a fresh `Python >=3.10` environment, e.g. via [Anaconda](https://www.anaconda.com/download):
```
conda create -n helder_env python=3.10
conda activate helder_env
```
Next, install a version of `PyTorch` that is compatible with your `CUDA` version (a list of `PyTorch`/`CUDA` combinations is available [here](https://pytorch.org/get-started/previous-versions/)), e.g.
```
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
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

## License
All files are provided under the terms of the BSD 2-Clause license.
