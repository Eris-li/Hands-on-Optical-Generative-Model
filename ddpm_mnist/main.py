import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

from .ddpm.model import DDPM
from .ddpm.trainer import DDPMTrainer
from .ddpm.sampler import DDPMSampler
from .classifier.model import MNISTClassifier
from .classifier.trainer import ClassifierTrainer
from .utils.data_loader import get_mnist_dataloader, get_train_val_loaders, get_test_loader
from .utils.helpers import save_image, load_image, set_seed, tensor_to_image, image_to_tensor


class MNISTSystem:
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 42,
    ):
        self.device = device
        set_seed(seed)
        
        self.ddpm_model = None
        self.ddpm_trainer = None
        self.ddpm_sampler = None
        
        self.classifier_model = None
        self.classifier_trainer = None
        
        self.is_ddpm_trained = False
        self.is_classifier_trained = False

    def init_ddpm(
        self,
        image_size: int = 28,
        hidden_channels: int = 64,
        time_emb_dim: int = 128,
    ):
        self.ddpm_model = DDPM(
            image_size=image_size,
            in_channels=1,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
        ).to(self.device)
        
        self.ddpm_optimizer = torch.optim.Adam(self.ddpm_model.parameters(), lr=1e-4)
        
        self.ddpm_trainer = DDPMTrainer(
            model=self.ddpm_model,
            optimizer=self.ddpm_optimizer,
            device=self.device,
            timesteps=1000,
        )
        
        self.ddpm_sampler = DDPMSampler(
            model=self.ddpm_model,
            trainer=self.ddpm_trainer,
            device=self.device,
        )

    def init_classifier(
        self,
        hidden_dims: list = [32, 64, 128, 256],
    ):
        self.classifier_model = MNISTClassifier(
            in_channels=1,
            num_classes=10,
            hidden_dims=hidden_dims,
        ).to(self.device)
        
        self.classifier_optimizer = torch.optim.Adam(self.classifier_model.parameters(), lr=1e-3)
        
        self.classifier_trainer = ClassifierTrainer(
            model=self.classifier_model,
            optimizer=self.classifier_optimizer,
            device=self.device,
        )

    def train_ddpm(
        self,
        train_loader,
        num_epochs: int = 20,
        save_path: str = 'checkpoints/ddpm_mnist.pth',
        log_interval: int = 100,
    ):
        if self.ddpm_model is None:
            self.init_ddpm()
        
        print(f"Training DDPM on {self.device}...")
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                
                loss = self.ddpm_trainer.train_step(images)
                total_loss += loss
                num_batches += 1
                
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss:.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                self.save_ddpm(save_path.replace('.pth', f'_epoch{epoch+1}.pth'))
        
        self.save_ddpm(save_path)
        self.is_ddpm_trained = True
        print("DDPM training completed!")

    def train_classifier(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        save_path: str = 'checkpoints/classifier_mnist.pth',
    ):
        if self.classifier_model is None:
            self.init_classifier()
        
        print(f"Training Classifier on {self.device}...")
        
        self.classifier_trainer.train(train_loader, val_loader, num_epochs)
        
        self.save_classifier(save_path)
        self.is_classifier_trained = True
        print("Classifier training completed!")

    def generate_digit(self, num_samples: int = 1, seed: int = None) -> torch.Tensor:
        if self.ddpm_sampler is None:
            raise RuntimeError("DDPM not initialized. Call init_ddpm() or load_ddpm() first.")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        samples = self.ddpm_sampler.sample(
            batch_size=num_samples,
            image_size=28,
            channels=1,
        )
        
        return samples

    def generate_and_save(
        self,
        num_samples: int = 16,
        save_path: str = 'outputs/generated_samples.png',
        nrow: int = 4,
    ):
        samples = self.generate_digit(num_samples)
        
        samples_denorm = (samples + 1) / 2
        
        save_image(
            samples_denorm,
            save_path,
            nrow=nrow,
            normalize=False,
        )
        
        print(f"Generated samples saved to {save_path}")
        return samples

    def recognize_digit(self, image_input) -> tuple:
        if self.classifier_model is None:
            raise RuntimeError("Classifier not initialized. Call init_classifier() or load_classifier() first.")
        
        if isinstance(image_input, str):
            image_tensor = load_image(image_input, size=(28, 28))
        elif isinstance(image_input, Image.Image):
            image_tensor = image_to_tensor(image_input)
        elif isinstance(image_input, torch.Tensor):
            image_tensor = image_input
        else:
            raise ValueError("Unsupported input type. Use path (str), PIL.Image, or torch.Tensor.")
        
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        predictions, probs = self.classifier_model.predict(image_tensor)
        
        predicted_digit = predictions.item()
        confidence = probs[0, predicted_digit].item()
        
        all_probs = {i: probs[0, i].item() for i in range(10)}
        
        return predicted_digit, confidence, all_probs

    def recognize_batch(self, images: torch.Tensor) -> tuple:
        if self.classifier_model is None:
            raise RuntimeError("Classifier not initialized. Call init_classifier() or load_classifier() first.")
        
        images = images.to(self.device)
        
        probs = self.classifier_model.predict_proba(images)
        predictions = torch.argmax(probs, dim=1)
        
        confidences = probs.gather(1, predictions.unsqueeze(1)).squeeze()
        
        return predictions.cpu(), confidences.cpu(), probs.cpu()

    def recognize_and_show(self, image_input) -> dict:
        predicted_digit, confidence, all_probs = self.recognize_digit(image_input)
        
        print(f"Recognized Digit: {predicted_digit}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("All probabilities:")
        for digit, prob in sorted(all_probs.items()):
            bar = '█' * int(prob * 20)
            print(f"  {digit}: {prob:.4f} {bar}")
        
        return {
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probs,
        }

    def save_ddpm(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.ddpm_trainer.save(path)
        print(f"DDPM model saved to {path}")

    def load_ddpm(self, path: str):
        if self.ddpm_model is None:
            self.init_ddpm()
        
        self.ddpm_trainer.load(path)
        self.ddpm_sampler = DDPMSampler(
            model=self.ddpm_model,
            trainer=self.ddpm_trainer,
            device=self.device,
        )
        self.is_ddpm_trained = True
        print(f"DDPM model loaded from {path}")

    def save_classifier(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.classifier_trainer.save(path)
        print(f"Classifier model saved to {path}")

    def load_classifier(self, path: str):
        if self.classifier_model is None:
            self.init_classifier()
        
        self.classifier_trainer.load(path)
        self.is_classifier_trained = True
        print(f"Classifier model loaded from {path}")

    def load_all(self, ddpm_path: str = None, classifier_path: str = None):
        if ddpm_path:
            self.load_ddpm(ddpm_path)
        if classifier_path:
            self.load_classifier(classifier_path)
