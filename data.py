import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List

class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for Maritime Trajectory Data.
    
    Expected CSV Columns:
    - ID Column (e.g., 'LLI NO')
    - Time Column (e.g., 'Date/Time')
    - Lat, Lng (Position)
    - SOG, COG (Navigation)
    - VMDR, VHM0 (Wave Data)
    """
    def __init__(self, df: pd.DataFrame, 
                 id_col='LLI NO', time_col='Date/Time', 
                 lat_col='Lat', lon_col='Lng', 
                 sog_col='SOG', cog_col='COG', 
                 wave_dir_col='VMDR', wave_height_col='VHM0',
                 groups: List[str] = None,
                 pos_mean=None, pos_std=None,
                 nav_mean=None, nav_std=None,
                 wave_mean=None, wave_std=None):
        
        # Data Loading
        self.id_col = id_col
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        # Group by Vessel ID
        self.groups_df = {gid: g.sort_values(time_col) for gid, g in df.groupby(id_col)}
        self.group_ids = sorted(self.groups_df.keys()) if groups is None else groups

        X_pos_list, X_nav_list, X_wave_list = [], [], []
        
        for gid in self.group_ids:
            g = self.groups_df[gid]
            
            # Extract features as numpy arrays
            pos = g[[lat_col, lon_col]].to_numpy(dtype=np.float32)
            nav = g[[sog_col, cog_col]].to_numpy(dtype=np.float32)
            wave = g[[wave_height_col, wave_dir_col]].to_numpy(dtype=np.float32)
            
            # Length Check (Critical for Batch Processing)
            # Assuming fixed sequence length of 110 for this model structure
            if pos.shape[0] != 110:
                continue
                
            X_pos_list.append(pos)
            X_nav_list.append(nav)
            X_wave_list.append(wave)

        self.X_pos = np.stack(X_pos_list)
        self.X_nav = np.stack(X_nav_list)
        self.X_wave = np.stack(X_wave_list)

        # Normalization (Z-Score)
        # If mean/std are not provided (Training mode), calculate them.
        if pos_mean is None:
            self.pos_mean = self.X_pos.reshape(-1, 2).mean(axis=0)
            self.pos_std = self.X_pos.reshape(-1, 2).std(axis=0) + 1e-8
        else:
            self.pos_mean, self.pos_std = pos_mean, pos_std

        if nav_mean is None:
            self.nav_mean = self.X_nav.reshape(-1, 2).mean(axis=0)
            self.nav_std = self.X_nav.reshape(-1, 2).std(axis=0) + 1e-8
        else:
            self.nav_mean, self.nav_std = nav_mean, nav_std

        if wave_mean is None:
            self.wave_mean = self.X_wave.reshape(-1, 2).mean(axis=0)
            self.wave_std = self.X_wave.reshape(-1, 2).std(axis=0) + 1e-8
        else:
            self.wave_mean, self.wave_std = wave_mean, wave_std

        # Apply Normalization
        self.X_pos_norm = (self.X_pos - self.pos_mean) / self.pos_std
        self.X_nav_norm = (self.X_nav - self.nav_mean) / self.nav_std
        self.X_wave_norm = (self.X_wave - self.wave_mean) / self.wave_std

    def __len__(self):
        return len(self.X_pos)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_pos_norm[idx]),
            torch.from_numpy(self.X_nav_norm[idx]),
            torch.from_numpy(self.X_wave_norm[idx]),
            torch.from_numpy(self.X_pos_norm[idx])
        )

def split_ids_by_group(df, id_col='LLI NO', train_ratio=0.9):
    """Randomly splits vessel IDs into training and validation sets."""
    gids = list(df[id_col].unique())
    import random
    random.shuffle(gids)
    n_train = int(len(gids) * train_ratio)
    return gids[:n_train], gids[n_train:]