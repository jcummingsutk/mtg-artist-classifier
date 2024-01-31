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


def build_classifier(base_model, num_classes: int):
    fully_connected = FullyConnectedLayer(base_model, num_classes)
    for param in base_model.parameters():
        param.requires_grad = False
    fully_connected = FullyConnectedLayer(base_model, num_classes)
    base_model.fc = fully_connected

    return base_model
