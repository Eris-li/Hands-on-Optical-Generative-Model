import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(
    batch_size: int = 32,
    train: bool = True,
    download: bool = True,
    num_workers: int = 2,
    shuffle: bool = True,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=train,
        transform=transform,
        download=download,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def get_train_val_loaders(
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 2,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    full_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True,
    )
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def get_test_loader(batch_size: int = 32, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader
