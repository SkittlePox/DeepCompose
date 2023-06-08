### This file contains the propositional primitives. These are learned functions from raw input to boolean values.
import torch
import torch.nn as nn

class PropositionalPrimitive(nn.Module):
    """
    This is a primitive classifier from images to truth values.
    """

    def __init__(self, image_dims, image_channels=3):
        self.semantics = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 20 * 30, 256),
            nn.ReLU(),
            nn.Linear(256, 96)
        )

    def __forward__(self, x):
        return self.semantics(x)
