### This file contains the propositional primitives. These are learned functions from raw input to boolean values.
import torch
import torch.nn as nn

class OldPropositionalPrimitive(nn.Module):
    """
    This is a primitive classifier from images to truth values.
    """

    def __init__(self):
        super().__init__()
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

    def forward(self, x):
        return self.semantics(x)

class PropositionalPrimitive(nn.Module):
    """Just like OldPropositionalPrimitive, but it takes in a 60 by 60 grayscale image and outputs a 1 dimensional output."""

    def __init__(self, digit, architecture="smallnet"):
        super().__init__()

        if architecture == "beefy":
            self.semantics = nn.Sequential(       # These may be too powerful - beefy
                nn.Conv2d(1, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(13 * 13 * 64, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        elif architecture == "smallnet":
            self.semantics = nn.Sequential(       # These are weak - smallnet
                nn.Conv2d(1, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(29 * 29 * 32, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        elif architecture == "lenet":
            self.semantics = nn.Sequential(         # This is almost LeNet - lenet
                nn.Conv2d(1, 6, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(16 * 12 * 12, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 1),
                nn.Sigmoid()
            )

        self.digit = digit

    def forward(self, x):
        return self.semantics(x)

    def __str__(self):
        return f"PropositionalPrimitive({self.digit})"

class PropositionPrimitiveCollection(nn.Module):
    """A bunch of PropositionalPrimitives put into one module"""

    def __init__(self, primitives):
        super().__init__()
        self.semantics = nn.ModuleList(primitives)
    
    def forward(self, x):
        return torch.cat([p(x) for p in self.semantics], dim=1)
