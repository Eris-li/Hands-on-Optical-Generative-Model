"""
MNIST DDPM System Usage Examples

This module provides examples of how to use the MNIST DDPM system for:
1. Generating handwritten digits
2. Recognizing handwritten digits
"""

import torch
from ddpm_mnist import MNISTSystem, get_mnist_dataloader, set_seed


def example_generate_digit():
    """Example: Generate handwritten digits"""
    set_seed(42)
    
    system = MNISTSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    system.init_ddpm()
    
    system.load_ddpm('checkpoints/ddpm_mnist.pth')
    
    samples = system.generate_digit(num_samples=16)
    print(f"Generated {samples.shape[0]} samples with shape {samples.shape}")
    
    system.generate_and_save(
        num_samples=16,
        save_path='outputs/generated_digits.png',
        nrow=4
    )


def example_recognize_digit():
    """Example: Recognize a handwritten digit from image file"""
    system = MNISTSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    system.init_classifier()
    
    system.load_classifier('checkpoints/classifier_mnist.pth')
    
    result = system.recognize_and_show('path/to/your/digit_image.png')


def example_training_ddpm():
    """Example: Train the DDPM model"""
    set_seed(42)
    
    train_loader = get_mnist_dataloader(batch_size=64, train=True, download=True)
    
    system = MNISTSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    system.init_ddpm(hidden_channels=64)
    
    system.train_ddpm(
        train_loader=train_loader,
        num_epochs=20,
        save_path='checkpoints/ddpm_mnist.pth',
        log_interval=100
    )


def example_training_classifier():
    """Example: Train the classifier model"""
    set_seed(42)
    
    train_loader = get_mnist_dataloader(batch_size=64, train=True, download=True)
    test_loader = get_mnist_dataloader(batch_size=64, train=False, download=True)
    
    system = MNISTSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    system.init_classifier()
    
    system.train_classifier(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=10,
        save_path='checkpoints/classifier_mnist.pth'
    )


def example_full_workflow():
    """Example: Full workflow from training to inference"""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 50)
    print("Full MNIST DDPM System Workflow")
    print("=" * 50)
    
    train_loader = get_mnist_dataloader(batch_size=64, train=True, download=True)
    test_loader = get_mnist_dataloader(batch_size=64, train=False, download=True)
    
    system = MNISTSystem(device=device, seed=42)
    
    print("\n[1] Initializing models...")
    system.init_ddpm(hidden_channels=32)
    system.init_classifier(hidden_dims=[32, 64, 128])
    
    print("\n[2] Training DDPM (for demonstration, using fewer epochs)...")
    system.train_ddpm(
        train_loader=train_loader,
        num_epochs=2,
        save_path='checkpoints/ddpm_demo.pth'
    )
    
    print("\n[3] Training Classifier...")
    system.train_classifier(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=2,
        save_path='checkpoints/classifier_demo.pth'
    )
    
    print("\n[4] Generating samples...")
    system.generate_and_save(
        num_samples=4,
        save_path='outputs/demo_generated.png',
        nrow=2
    )
    
    print("\n[5] Testing recognition on real MNIST samples...")
    for images, labels in test_loader:
        for i in range(min(3, len(images))):
            img = images[i]
            true_label = labels[i].item()
            
            pred_label, conf, probs = system.recognize_digit(img)
            
            print(f"Sample {i+1}: True={true_label}, Pred={pred_label}, Conf={conf:.2%}")
        
        break
    
    print("\n" + "=" * 50)
    print("Workflow completed!")
    print("=" * 50)


if __name__ == '__main__':
    example_full_workflow()
