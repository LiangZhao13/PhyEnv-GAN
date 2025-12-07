# Trajectory cGAN with Physics-Informed Motion Constraints & Wave Conditioning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)](https://pytorch.org/)

A PyTorch implementation of a **Conditional Generative Adversarial Network (cGAN)** for maritime vessel trajectory reconstruction and generation. 

This research integrates environmental factors into trajectory modeling. It features:
1.  **Deep Learning Core:** An LSTM-based CVAE Generator paired with a TCN (Temporal Convolutional Network) Discriminator.
2.  **Physics-Informed:** Incorporates a kinematic motion-model constraint (PINN) to ensure physically feasible trajectories.
3.  **Wave Conditioning:** Explicitly conditions the generation process on wave height (`VHM0`) and direction (`VMDR`).

## 📂 Project Structure

```text
.
├── main.py        # Entry point: Data validation and training loop
├── models.py      # Neural Network architectures (CVAE, TCN Discriminator)
├── data.py        # Dataset loading and preprocessing logic
├── utils.py       # Math tools (Haversine, DTW) and Physics Loss functions
└── README.md      # Project documentation
