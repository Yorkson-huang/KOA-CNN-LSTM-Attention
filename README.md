<h1 align="center">KOA-CNN-LSTM-Attention for Wind Speed Prediction</h1>

<p align="center">
  <a href="https://www.mathworks.com/products/matlab.html"><img src="https://img.shields.io/badge/MATLAB-2023a-blue.svg" alt="MATLAB"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-yellow.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <a href="https://www.engineeringletters.com/issues_v32/issue_10/EL_32_10_14.pdf"><img src="https://img.shields.io/badge/Paper-Engineering%20Letters-8A2BE2.svg" alt="Paper"></a>
</p>

---

## Table of Contents
- [Overview](#overview)
- [MATLAB Implementation](#matlab-implementation)
- [Python Port (Experimental)](#python-port-experimental)
- [Dataset Structure](#dataset-structure)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview
This repository accompanies the paper "An Improved Hybrid CNN-LSTM-Attention Model with Kepler Optimization Algorithm for Wind Speed Prediction." The original implementation is provided in MATLAB and combines convolutional neural networks, LSTM layers, an attention mechanism, and the Kepler Optimization Algorithm (KOA) for wind speed forecasting.

---

## MATLAB Implementation
- **Core scripts:**
  - `MAIN.m` - builds and trains the CNN-LSTM-Attention network.
  - `objectiveFunction.m` - defines the loss used by KOA.
  - `KOA.m` - implements the Kepler Optimization Algorithm.
- **Requirements:** MATLAB 2023a (or later) with the Deep Learning Toolbox; the Optimization Toolbox is recommended for KOA experiments.
- **How to run:**
  1. Place the Excel dataset (e.g., `Data.xlsx`; legacy name `FeatureSequenceAndActual.xlsx`) in the project root.
  2. Open MATLAB, change to this directory, and run `MAIN.m` for a single training run.
  3. Run `KOA.m` to launch hyper-parameter optimization.

> MATLAB scripts represent the authoritative version of the method described in the paper.

---

## Python Port (Experimental)
A PyTorch-based port lives in the `python_impl/` folder. It mirrors the MATLAB workflow to support quick experimentation and scripting.

- **Key files:**
  - `main.py` - command-line entry point.
  - `objective_function.py` - training loop equivalent to `objectiveFunction.m`.
  - `koa.py` - KOA translated to Python.
  - `model.py` - CNN-LSTM-Attention architecture.
  - `data_utils.py` - Excel data loader.
- **Setup:**
  ```bash
  pip install -r python_impl/requirements.txt
  python -m python_impl.main --data-path python_impl/Data.xlsx
  ```
- **KOA example:**
  ```bash
  python -m python_impl.main \
    --use-koa --agents 10 --iterations 30 \
    --data-path python_impl/Data.xlsx
  ```

The Python version is provided for convenience and experimentation alongside the MATLAB code.

---

## Dataset Structure
- Single worksheet; the used range contains 20 rows by 1801 columns.
- Column A stores row labels:
  - `feature1` ... `feature18` - 18 feature sequences.
  - `target` - wind-speed ground truth.
- Columns B through FZ (1,800 columns) are labelled `time1` ... `time1800`, representing 75 days x 24 hours (hourly resolution).

---

## Citation
If you use this work, please cite the original paper:

```bibtex
@article{huang2024improved,
  title={An Improved Hybrid CNN-LSTM-Attention Model with Kepler Optimization Algorithm for Wind Speed Prediction},
  author={Huang, Yuesheng and Li, Jiawen and Li, Yushan and Lin, Routing and Wu, Jingru and Wang, Leijun and Chen, Rongjun},
  journal={Engineering Letters},
  volume={32},
  number={10},
  pages={1957--1965},
  year={2024},
  publisher={IAENG}
}
```

---

## Contact
For questions or collaborations, reach out to **yorksonhuang@gmail.com**.

---

<p align="center">Made with love for wind energy research.</p>
