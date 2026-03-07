import torch
import numpy as np
import random

# Earth Radius in meters (approximate)
R = 6371000.0

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Calculates the Great Circle Distance (in meters) between two points.
    """
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def dynamic_loss(y_fake, x_nav):
    """
    Physics-Informed Motion Constraint.
    Enforces: Pos(t+1) = Pos(t) + Velocity(t) * dt
    """
    SOG = x_nav[:, :, 0]
    COG = x_nav[:, :, 1] * np.pi / 180
    v_nav = torch.stack([SOG*torch.cos(COG), SOG*torch.sin(COG)], dim=-1)
    motion_res = y_fake[:, :-1] + v_nav[:, :-1] - y_fake[:, 1:]
    return torch.mean(torch.norm(motion_res, dim=-1))