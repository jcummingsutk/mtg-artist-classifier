import torch.nn as nn
import torch.nn.functional as f
import torchvision


class ArtistClassifier(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
