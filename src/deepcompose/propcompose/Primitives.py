### This file contains the propositional primitives. These are learned functions from raw input to boolean values.
import torch
import torch.nn as nn

class PropositionalPrimitive(nn.Module):
    """
    This is a primitive classifier from images to truth values.
    """

    def __init__(self, image_dims, image_channels=3):
        self.semantics = nn.Sequential(
            # TODO: Add convolutional layers for a 320 x 480 RGB image. Output a single boolean.
        )

    def __forward__(self, x):
        pass