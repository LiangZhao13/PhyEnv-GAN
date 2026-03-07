import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import local modules
from utils import set_seed, dynamic_loss
from models import CVAEGenerator, TCN_Discriminator
from data import TrajectoryDataset, split_ids_by_group

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
#  INSTRUCTION FOR USERS
# ==========================================
# Make sure your data aligns with the following structure, and each trajectory should contain exact 110 points, otherwise there will be bugs
REQUIRED_COLUMNS = [
    'LLI NO',      # Vessel ID, if you use IMO/MMSI number, just change the colunm name "IMO" or "MMSI" to "LLI NO", and it won't cause bugs
    'Date/Time',   # Timestamp
    'Lat', 'Lng',  # Position
    'SOG', 'COG',  # Navigation
    'VMDR', 'VHM0' # Wave (Direction, Height)
]

def check_data_availability(csv_path):
    """
    Checks if the dataset exists and validates the column structure.
    Returns True if valid, False otherwise.
    """
    if not os.path.exists(csv_path):
        print("\n" + "="*60)
        print(f" [ERROR] Dataset file not found: '{csv_path}'")
        print("="*60)
        print(" To run this model, you need to provide your own AIS data.")
        print(" Please prepare a CSV file with the following structure:")
        print("-" * 40)
        print(f" Columns required: {', '.join(REQUIRED_COLUMNS)}")
        print(" Sequence Length : Each vessel ID must have at least 110 data points.")
        print("-" * 40)
        print(" Example Row:")
        print(" LLI NO,   Date/Time,           Lat,    Lng,     SOG,  COG,   VMDR, VHM0")
        print(" 1234567,  2023-01-01 12:00,    34.56,  128.12,  12.5, 180.0, 45.0, 1.5")
        print("="*60 + "\n")
        return False

    # Optional: Quick header check
    try:
        df_head = pd.read_csv(csv_path, nrows=1)
        missing = [c for c in REQUIRED_COLUMNS if c not in df_head.columns]
        if missing:
            print(f"\n[ERROR] CSV is missing columns: {missing}\n")
            return False
    except Exception as e:
        print(f"\n[ERROR] Could not read CSV: {e}\n")
        return False

    return True


def run_training(
    csv_path='your_ais_data.csv',
    batch_size=32, 
    epochs=150,
    lr_G=1e-3, 
    lr_D=1e-3,
    out_dir='outputs'
):
    # 1. Pre-check: Ensure data exists
    if not check_data_availability(csv_path):
        return

    # 2. Setup
    set_seed(42)
    os.makedirs(os.path.join(out_dir, "ckpts"), exist_ok=True)
    
    print(f">> Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    train_ids, val_ids = split_ids_by_group(df)
    
    ds_train = TrajectoryDataset(df, groups=train_ids)
    ds_val = TrajectoryDataset(
        df, groups=val_ids,
        pos_mean=ds_train.pos_mean, pos_std=ds_train.pos_std,
        nav_mean=ds_train.nav_mean, nav_std=ds_train.nav_std,
        wave_mean=ds_train.wave_mean, wave_std=ds_train.wave_std
    )
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    
    print(f">> Data loaded. Train Ships: {len(train_ids)} | Val Ships: {len(val_ids)}")

    # 3. Initialize Models
    G = CVAEGenerator().to(DEVICE)
    D = TCN_Discriminator().to(DEVICE)
    
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_D)
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    
    print(">> Start Training...")
    
    # 4. Training Loop
    for epoch in range(1, epochs+1):
        G.train(); D.train()
        loss_G_list, loss_D_list = [], []

        for x_pos, x_nav, x_wave, y_pos in train_loader:
            x_pos, x_nav, x_wave, y_pos = [t.float().to(DEVICE) for t in [x_pos, x_nav, x_wave, y_pos]]
            B = x_pos.size(0)

            # --- Train Discriminator ---
            with torch.no_grad():
                y_fake, _, _ = G(x_pos, x_nav, x_wave)
            loss_D = 0.5 * (
                bce(D(x_pos, y_pos, x_wave), torch.ones(B,1,device=DEVICE)) + 
                bce(D(x_pos, y_fake, x_wave), torch.zeros(B,1,device=DEVICE))
            )
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # --- Train Generator ---
            y_fake, mu, logvar = G(x_pos, x_nav, x_wave)
            adv_loss = bce(D(x_pos, y_fake, x_wave), torch.ones(B,1,device=DEVICE))
            rec_loss = l1(y_fake, y_pos)
            dyn_loss = dynamic_loss(y_fake, x_nav)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss_G = adv_loss + 10.0*rec_loss + 1.0*dyn_loss + 0.001*kl_loss
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            
            loss_D_list.append(loss_D.item())
            loss_G_list.append(loss_G.item())

        # Validation Log
        print(f"[Epoch {epoch:03d}] Loss D: {np.mean(loss_D_list):.4f} | Loss G: {np.mean(loss_G_list):.4f}")
        
        # Save checkpoints periodically
        if epoch % 10 == 0:
            torch.save(G.state_dict(), os.path.join(out_dir, "ckpts", f"G_epoch_{epoch}.pt"))

if __name__ == "__main__":
    # You can change the path here to your own data file, one trajectory should contain exact 110 points
    run_training(csv_path='your_ais_data.csv')