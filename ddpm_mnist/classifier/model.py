import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dims: list = [32, 64, 128, 256],
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dims[1], hidden_dims[2], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dims[2], hidden_dims[3], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dims[3], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            return predictions, probs

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            return probs
