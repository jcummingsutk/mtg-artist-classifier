import torch.nn as nn


class ArtistClassifier(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
