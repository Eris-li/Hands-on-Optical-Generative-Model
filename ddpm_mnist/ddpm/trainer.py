import torch
import torch.nn as nn
from pathlib import Path


class DDPMTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        scheduler: str = 'cosine',
        timesteps: int = 1000,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.timesteps = timesteps

        if scheduler == 'cosine':
            self.betas = DDPM.cosine_beta_schedule(timesteps)
        else:
            self.betas = DDPM.linear_beta_schedule(timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(self, x, t, t_index):
        x_recon = self._predict_start_from_noise(
            x, t, self._predict_noise(x, t, t_index)
        )

        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * x_recon
            + self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        model_variance = self._extract(self.posterior_variance, t, x.shape)
        model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)

        return model_mean, model_variance, model_log_variance

    def _predict_start_from_noise(self, x_t, t, noise_pred):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise_pred
        )

    def _predict_noise(self, x, t, t_index=None):
        return self.model(x, t)

    def _extract(self, coefficients, t, x_shape):
        batch_size = t.shape[0]
        out = coefficients.to(t.device).gather(0, t)
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))

    def train_step(self, x):
        self.model.train()
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x)
        
        x_noisy = self.q_sample(x, t, noise)
        noise_pred = self.model(x_noisy, t)
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'betas': self.betas,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.betas = checkpoint['betas']
