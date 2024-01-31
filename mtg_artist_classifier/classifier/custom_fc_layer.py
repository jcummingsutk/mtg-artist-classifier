import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        num_ftrs = base_model.fc.in_features
        self.fully_connected = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fully_connected(x)
