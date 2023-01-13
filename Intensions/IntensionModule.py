import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class EntityModule(nn.Module):
    def __init__(self, ent_id, num_ents):
        super().__init__()
        self.intension = torch.zeros(num_ents)
        self.intension[ent_id] = 1.0

    def forward(self, x):
        return self.intension


class IntensionModule(nn.Module):
    def __init__(self, semantic_type, num_ents, hidden_dim=32, image_channels=3, image_dim=32):
        super().__init__()
        self.semantic_type = semantic_type

        self.intension = nn.Sequential(
            nn.Conv2d(image_channels, image_dim, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(1024, hidden_dim*num_ents)
        )

    def forward(self, x):
        return self.intension(x)


def create_entity_modules(entities):
    for e in entities:
        e.intension = EntityModule
