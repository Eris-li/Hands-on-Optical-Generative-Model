import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, channels)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class DDPM(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        hidden_channels: int = 64,
        time_emb_dim: int = 128,
        num_residual_blocks: int = 2,
        attention_levels: list = [2],
        num_layers: int = 3,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim

        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.first_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        ch = hidden_channels
        down_chs = [hidden_channels]
        
        for i in range(num_layers):
            for _ in range(num_residual_blocks):
                self.down_blocks.append(ResidualBlock(ch, time_emb_dim))
            
            if i < num_layers - 1:
                down_chs.append(ch * 2)
                self.down_blocks.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1))
                ch = ch * 2

        self.middle_block1 = ResidualBlock(ch, time_emb_dim)
        self.middle_attn = nn.MultiheadAttention(ch, 4, batch_first=True) if 0 in attention_levels else None
        self.middle_block2 = ResidualBlock(ch, time_emb_dim)

        for i in range(num_layers - 1, -1, -1):
            for _ in range(num_residual_blocks):
                self.up_blocks.append(ResidualBlock(ch, time_emb_dim))
            
            if i > 0:
                self.up_blocks.append(nn.Conv2d(ch, ch // 2, 4, stride=2, padding=1, transpose=True))
                ch = ch // 2
            down_chs.pop()

        self.final_norm = nn.GroupNorm(8, hidden_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def get_time_embedding(self, t):
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        return t_emb

    def forward(self, x, t):
        t_emb = self.get_time_embedding(t)
        
        h = self.first_conv(x)
        
        hs = []
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
                hs.append(h)
            else:
                h = block(h)
        
        if self.middle_attn is not None:
            b, c, w, h_ = h.shape
            h_flat = h.flatten(2).permute(0, 2, 1)
            h_flat, _ = self.middle_attn(h_flat, h_flat, h_flat)
            h = h_flat.permute(0, 2, 1).reshape(b, c, w, h_)
        
        h = self.middle_block1(h, t_emb)
        h = self.middle_block2(h, t_emb)
        
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)
        
        return h

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def linear_beta_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02):
        return torch.linspace(start, end, timesteps)
