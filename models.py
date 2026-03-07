import torch
import torch.nn as nn

class CVAEGenerator(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) Generator.
    """
    def __init__(self, in_pos_dim=2, in_nav_dim=2, in_wave_dim=2, 
                 hidden=128, num_layers=2, z_dim=16):
        super().__init__()
        # Encoder
        self.encoder = nn.LSTM(in_pos_dim + in_nav_dim + in_wave_dim, hidden, num_layers,
                               batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden*2, z_dim)
        self.fc_logvar = nn.Linear(hidden*2, z_dim)

        # Decoder
        dec_in = in_pos_dim + in_nav_dim + in_wave_dim + z_dim
        self.decoder = nn.LSTM(dec_in, hidden, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden, in_pos_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x_pos, x_nav, x_wave):
        B, T, _ = x_pos.shape
        enc_in = torch.cat([x_pos, x_nav, x_wave], dim=-1)
        _, (h, _) = self.encoder(enc_in)
        h_last = torch.cat([h[-2], h[-1]], dim=1)
        
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        z = self.reparameterize(mu, logvar)
        z_rep = z.unsqueeze(1).repeat(1, T, 1)

        dec_in = torch.cat([x_pos, x_nav, x_wave, z_rep], dim=-1)
        dec_out, _ = self.decoder(dec_in)
        return self.proj(dec_out), mu, logvar

class TCN_Discriminator(nn.Module):
    """
    Temporal Convolutional Network (TCN) Discriminator.
    """
    def __init__(self, in_dim=6, channels=[64, 128, 256], k=3):
        super().__init__()
        # Implementation details omitted for brevity (same as before)
        # ... (Include the full TemporalBlock and Chomp1d classes here)
        # Placeholder for brevity in this response:
        self.fc = nn.Linear(channels[-1], 1)
        # (Make sure to include the full code from previous answer here in actual file)
    
    def forward(self, x_pos, y_pos, x_wave):
        # Placeholder logic
        return torch.randn(x_pos.size(0), 1).to(x_pos.device)