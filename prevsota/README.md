# prevsota — Previous SOTA Baselines

Code taken from [pouliquen.25.icdar](https://github.com/EPITAResearchLab/pouliquen.25.icdar).

This directory contains the config files and modified evaluation script used to run the baselines under this paper's evaluation protocol.

## Contents

- `test.py`: modified evaluation script. Run it from the **repository** (`pouliquen.25.icdar`) using these config files, do not run it standalone from this directory.
- `conf/`: Hydra config files for each evaluated setting:
  - `experiment/valid_nonvalid/` — configs for the Legit+Non-Legit protocol (15 fps, 5 fps)
  - `experiment/validonly/` — configs for the Legit-only protocol

## Usage

1. Clone and set up [pouliquen.25.icdar](https://github.com/EPITAResearchLab/pouliquen.25.icdar) following its own README.
2. Copy (or symlink) `prevsota/conf/` into the companion repo's config directory, and replace its `test.py` with `prevsota/test.py`.
3. Run evaluation for a given setting, e.g.:

```bash
# from inside the pouliquen.25.icdar repo
python train.py --config-name=wsl -m +experiment=wsl/mobilevit_s_5fps_onlyorigins_old2_norota_icdar26 "paths.split_name=k0,k1,k2,k3,k4"

python calibration.py --config-name=wsl -m +experiment=wsl/mobilevit_s_5fps_onlyorigins_old2_norota_icdar26 "paths.split_name=k0,k1,k2,k3,k4" "decision=allvideo"

# with provided test.py
python test.py --config-name=wsl -m +experiment=wsl/mobilevit_s_5fps_onlyorigins_old2_norota_icdar26 "paths.split_name=k0,k1,k2,k3,k4" "decision=allvideo"
```
