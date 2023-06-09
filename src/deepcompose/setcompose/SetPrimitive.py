### This file contains set-theoretic primitives. There are two types of primitives: nouns and adjectives, each with different types.
### Nouns are learned functions from raw input to a set of objects, i.e. a set of bounding boxes.
### Adjectives are learned functions from bounding boxes to booleans.
### "big red cube" is implemented as cube;N big;A red;A >> big(cube) & red(cube) -> {t,f}

import torch
import torch.nn as nn


def blacken_outside(img, x, y, h, w):
    """
    Blacken out the area outside of the mask.
    """
    img = img.copy()
    img[:, :y, :] = 0
    img[:, y + h:, :] = 0
    img[:, y:y + h, :x] = 0
    img[:, y:y + h, x + w:] = 0
    return img


class SetAdjPrimitive(nn.Module):
    """
    This is a primitive classifier from images with bounding boxes to truth values.
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
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.semantics(x)
