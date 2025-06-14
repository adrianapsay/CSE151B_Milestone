# Climate Variable Emulation with Deep Learning

This repository contains the code and models developed for our final project in **CSE 151B: Deep Learning**, where we aim to emulate climate variables—specifically **temperature (tas)** and **precipitation (pr)**—using data-driven methods trained on CMIP6 climate model simulations.

## Project Structure

- `notebooks/` – Jupyter notebooks for experimentation and exploratory analysis  
- `models/` – Contains all model architecture implementations  
  - `all_models.py` – Unified script with all model variants (U-Net, SE, CoordConv, etc.) and documentation  
- `figures/` – Visualizations and plots used in reports and presentations  
- `core/` – Scripts for data preprocessing, descriptive statistics, and helpers  
- `README.md` – Project overview and usage instructions  
- `.gitignore` – File exclusion rules for checkpoints, metadata, and cache

## Goal

Predict climate variables **tas** and **pr** one month at a time from historical and SSP scenario data using supervised learning. Evaluate performance with area-weighted **RMSE** and **MAE**, focusing on spatial generalization and temporal robustness.

## Models

We developed and compared several models with increasing complexity:

- **SimpleMLP** – A baseline fully connected model without spatial structure
- **SimpleCNN** – A shallow CNN with residual blocks
- **UNetCNN (v3)** – A U-Net with CoordConv and shared output head
- **UNetCNN + SE** – U-NetCNN with Squeeze-and-Excitation blocks
- **Improved UNet** – U-NetCNN + SE + Attention Gates (discarded due to instability)

### Best Model: `UNetCNN (CoordConv + Shared Output)`

This architecture uses:
- CoordConv inputs for spatial awareness
- Two encoder/decoder layers with skip connections
- Dropout (0.2) for regularization
- A **shared output head** for tas and pr

## Results

Our best-performing model on the **Kaggle private test set** garnered these metrics on the validation set:

| Metric         | tas RMSE | tas MAE | tas Corr | pr RMSE | pr MAE | pr Corr |
|----------------|----------|---------|----------|---------|--------|---------|
| **Score**      | 1.3437   | 0.5271  | 0.2522   | 1.9586  | 0.2952 | 0.7627  |

- **Final Leaderboard Score**: **0.9800 RMSE**
- **Leaderboard Rank**: **33 / 83**

## Training Details

- **Epochs**: 175 max with early stopping (patience=20)
- **Batch Size**: 64
- **Optimizer**: Adam
- **Scheduler**: CosineAnnealingLR
- **Loss**: MSE with equal weighting for tas and pr
- **Early Stopping Metric**: val/tas/rmse

## Key Learnings

- Simple architectures like CoordConv-based U-Nets generalize better than deeper, more complex variants.
- Empirical testing is critical—common techniques (e.g., log-transform, SE/attention modules) often underperform without proper tuning.
- Shared decoder heads offered better gradient flow than dual-head decoders, especially under time constraints.

## Future Work

- Incorporate temporal models (e.g., ConvLSTM or Transformer) for sequence prediction.
- Explore uncertainty-aware forecasting (e.g., quantile loss).
- Extend the pipeline for multi-step forecasting beyond one-month prediction.
