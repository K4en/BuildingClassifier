import torch
import torch.nn as nn

class BuildingClassifier(nn.Module):
    def __init__(self):
        super(BuildingClassifier, self).__init__()

        # Feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # 16x128x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 16x64x64

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), #output 32x64x64
            nn.ReLU(),
            nn.MaxPool2d(2), # output 32x32x32

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 64x32x32
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x16x16
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(), # flatten for linear layer input
            nn.Linear(in_features=64 * 16 * 16, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3), # prevent over fitting
            nn.Linear(128, 1), # Binary output
            nn.Sigmoid() # Squash output to [0, 1] for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x) # Extract features
        x = self.fc_layers(x)   # Classify
        return x