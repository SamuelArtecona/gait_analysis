# Gait Analysis:
Repository aims to analyze emergent robot gaits trained via DRL and make them more biologically relevant.
This repository contains a slightly modified and stripped version of Brax's `training` utilities.

## Installation Requirements:
It is recommended you have `python3.12` or newer. This repository uses `wandb` for logging and requires an account.

```
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install jax
pip install brax
pip install flax
pip install optax
pip install distrax
pip install orbax-checkpoint
```

If you want to log via `wandb`
```
pip install wandb
```

If you want to visualize intermediate training iterations
```
pip install opencv-python
```

## Debugging installation:
If you are having issues installing the above make sure you have all the required python development packages. (Optional: add deadsnakes repository for newest python releases `sudo add-apt-repository ppa:deadsnakes/ppa`)

```
sudo apt update
sudo apt install python3.12-full python3.12-dev
```
