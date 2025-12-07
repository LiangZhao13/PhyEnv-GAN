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

## 📊 Data Requirements

To run this model, you need to prepare your own AIS dataset. The code includes a strict validation step to ensure the data format matches the model's input requirements.

### 1. CSV Structure
Prepare a CSV file (default name: `your_ais_data.csv`) with the **exact** column names listed below. 

| Column Name | Type | Description | Unit / Example |
| :--- | :--- | :--- | :--- |
| **LLI NO** | String/Int | Unique Vessel ID (MMSI or IMO) | `123456789` |
| **Date/Time** | String | Timestamp | `2023-01-01 12:00:00` |
| **Lat** | Float | Latitude | `34.1234` |
| **Lng** | Float | Longitude | `128.5678` |
| **SOG** | Float | Speed Over Ground | Knots (e.g., `12.5`) |
| **COG** | Float | Course Over Ground | Degrees (0-360) |
| **VMDR** | Float | Mean Wave Direction | Degrees |
| **VHM0** | Float | Significant Wave Height | Meters |

> **⚠️ Important:** If your raw data uses different headers (e.g., "MMSI", "Speed", "WaveHeight"), you **must rename them** to match the names above, or the script will raise an error.

### 2. Sequence Length Constraint
The model architecture requires fixed-length input tensors. 
* **Constraint:** Each vessel trajectory (grouped by `LLI NO`) must contain **exactly 110 data points**.
* **Preprocessing:** Please interpolate or segment your raw trajectories to length 110 before training.
* **Automatic Filtering:** The data loader will automatically skip any vessel ID that does not meet this length requirement.

---

## 🚀 How to Run (`main.py`)

The `main.py` script handles data validation, model initialization, and the complete training loop.

### Step 1: Place Data
Put your formatted CSV file (e.g., `your_ais_data.csv`) in the root directory of the project.

### Step 2: Configure (Optional)
Open `main.py` to adjust hyperparameters if needed:

```python
if __name__ == "__main__":
    run_training(
        csv_path='your_ais_data.csv',  # Path to your dataset
        batch_size=32,                 # Batch size
        epochs=150,                    # Training epochs
        lr_G=1e-3,                     # Generator learning rate
        out_dir='outputs'              # Output directory
    )
