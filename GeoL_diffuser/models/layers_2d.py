from torch import nn
from einops.layers.torch import Rearrange
from GeoL_net.models.GeoL import FeaturePerceiver


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class DownsampleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),  
            nn.Mish(),            
            nn.Linear(dim, dim)   
        )

    def forward(self, x):
        batch, horizon, dim = x.shape
        assert horizon % 2 == 0, "horizon must be divisible by 2 for reduction."
        
        # Apply MLP on each timestep independently
        x = self.mlp(x)
        
        # Reduce the horizon dimension by averaging every two steps
        x = x.view(batch, horizon // 2, 2, dim).mean(dim=2)  # [batch, horizon/2, dim]
        return x

class UpsampleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),  
            nn.Mish(),            
            nn.Linear(dim, dim)   
        )

    def forward(self, x):
        batch, horizon, dim = x.shape
        
        # Upsample by repeating each timestep
        x = x.unsqueeze(2).repeat(1, 1, 2, 1).view(batch, horizon * 2, dim)  # [batch, horizon*2, dim]
        
        # Apply MLP on each timestep independently
        x = self.mlp(x)
        return x


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class PerceiverBlock(nn.Module):
    def __init__(
            self, 
            transition_dim, 
            condition_dim,
            time_emb_dim,
            decoder_q_input_channels):
        super().__init__()

        self.perceiver = FeaturePerceiver(
            transition_dim=transition_dim, 
            condition_dim=condition_dim,
            time_emb_dim=time_emb_dim,
            decoder_q_input_channels=decoder_q_input_channels
)
    
    def forward(self, x, condition_feat, time_embedding=None):
        """
        x: [batch, horizon, transition_dim]
        condition_feat: [batch, 1, condition_dim]
        time_embedding: [batch, 1, time_embedding_dim]

        """
        return self.perceiver(x, condition_feat, time_embedding) #[batch, horizon, decoder_q_input_channels]
        


