import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class DDPMSampler:
    def __init__(self, model: nn.Module, trainer, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.trainer = trainer
        self.timesteps = trainer.timesteps

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        image_size: int = 28,
        channels: int = 1,
        return_intermediates: bool = False,
    ):
        self.model.eval()
        
        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        
        intermediates = [] if return_intermediates else None
        
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            noise_pred = self.model(x, t)
            
            alpha = self.trainer._extract(self.trainer.alphas, t, x.shape)
            alpha_cumprod = self.trainer._extract(self.trainer.alphas_cumprod, t, x.shape)
            beta = self.trainer._extract(self.trainer.betas, t, x.shape)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
                ) + torch.sqrt(beta) * noise
            else:
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
                )
            
            if return_intermediates and i % 100 == 0:
                intermediates.append(x.cpu())
        
        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def generate_digit(self, digit: int, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        
        return self.sample(batch_size=1)

    @torch.no_grad()
    def generate_multiple_digits(self, num_samples: int = 10):
        samples = []
        for _ in range(num_samples):
            sample = self.sample(batch_size=1)
            samples.append(sample)
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def load_weights(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
