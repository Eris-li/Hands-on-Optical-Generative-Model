import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image
from pathlib import Path


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_image(tensor, path: str, nrow: int = 8, normalize: bool = True, value_range: tuple = (-1, 1)):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    if tensor.shape[1] in [1, 3] and tensor.shape[1] != 1:
        tensor = tensor.permute(0, 2, 3, 1)
    
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
    
    if grid.shape[0] == 1:
        grid = grid.squeeze(0)
    
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    img = Image.fromarray(ndarr)
    img.save(path)


def load_image(path: str, size: tuple = None) -> torch.Tensor:
    img = Image.open(path).convert('L')
    
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32)
    
    img_tensor = torch.from_numpy(img_array) / 255.0
    
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)
    
    img_tensor = img_tensor * 2 - 1
    
    return img_tensor


def tensor_to_image(tensor) -> Image.Image:
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    
    return Image.fromarray(ndarr)


def image_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    img_array = np.array(image.convert('L'), dtype=np.float32)
    
    if normalize:
        img_tensor = torch.from_numpy(img_array) / 255.0
        img_tensor = img_tensor * 2 - 1
    else:
        img_tensor = torch.from_numpy(img_array) / 255.0
    
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor
