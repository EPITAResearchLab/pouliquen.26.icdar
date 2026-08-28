# Temporal Modeling of Optically Variable Devices in Identity Documents
Repository for the paper "Temporal Modeling of Optically Variable Devices in Identity Documents" submitted to ICDAR 2026.


![introfigure](figures/IntroFigure.pdf)

---

## Repository Structure

The repository is organized into three main subdirectories:

- `prevsota/` – Previous state-of-the-art baselines, including `test.py`
- `sequenceclassif/` – Sequence classification experiments
- `msm/` – Masked Sequence Modeling experiments

---

## Getting Started

### 1. Baseline Models (Previous SOTA)

Run model training using the companion repository [pouliquen.25.icdar](https://github.com/EPITAResearchLab/pouliquen.25.icdar) for the following settings:

- MIDV-Holo Baseline (Legit and Non-Legit)
- HoloVerif (Legit and Non-Legit)
- WSL (Legit and Non-Legit)
- HoloVerif (Legit only)
- WSL (Legit only)

### 2. HoloVerif Span – Sequence Classification

```bash
sequenceclassif/jobs/main.sh
```

### 3. Masked Sequence Modeling

Specify the path to the pre-trained WSL models in the relevant config files, then run:

```bash
msm/jobs/main.sh
```

> **Note:** Ensure the WSL model paths are correctly set in the config files before launching the job.

---

## Retrieving and Visualizing Results

Use the following scripts to reproduce the results reported in the paper:

- `scripts/results/plot_roc.py` – Generate ROC curves
- `scripts/results/compare_methods_full.py` – Generate the main comparison table
- `scripts/results/export_latex.py` – Gather results for **Table 2** (LaTeX export)

## Citation

If you use this work, please cite:

```
@inproceedings{pouliquen_ovdtemporal_2026,
  TITLE = {{Temporal Modeling of Optically Variable Devices in Identity Documents}},
  AUTHOR = {Pouliquen, Glen and Chazalon, Joseph and Chiron, Guillaume and Ramos Terraded, Oriol and G{\'e}raud, Thierry and Awal, Ahmad Montaser},
}
```