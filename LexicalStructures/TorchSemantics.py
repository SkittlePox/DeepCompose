import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.io import read_image
from torchvision import transforms
from functools import reduce


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)


class ExtensionModule(nn.Module):
    """
    Every item in the lexicon has an ExtensionModule that goes from a state to a matrix, where the
    dimensions of the matrix corresponds to the semantic type of the lexical entry.

    For example: any noun phrase (NP) has an ExtensionModule that goes from states to matrices of
    size M x 1 or M. Likewise, an S/NP like 'grunts' has an ExtensionModule that goes from states to
    matrices of size 1 x M. This way, when you calculate the extension for a given world, you
    do a dot product on these matrices: 1 x M dot M x 1 = 1 x 1, ending up with a truth value
    between 0 and 1. For (S/NP)/NP you get a tensor that has dimensions 1 x M x M. (squeeze(-1))

    hidden_dim is M here.
    """

    def __init__(self, output_dims, image_channels=3, image_dim=32):
        super().__init__()

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
            nn.Linear(1024, reduce(lambda x, y: x * y, output_dims)),
            UnFlatten(dims=output_dims)
        )

    def forward(self, state):
        return self.intension(state)


class SemanticIntension(nn.Module):
    def __init__(self, name, module):
        super().__init__()
        self.name = name
        self.module = module

    def forward(self, state):
        return self.module.forward(state)

    def __str__(self):
        return f"Intension[{self.name}]"


class SemanticIntensionApplication(nn.Module):
    def __init__(self, function_module=None, argument_module=None):
        super().__init__()
        self.function_module = function_module
        self.argument_module = argument_module
        self.name = f"{self.function_module.name}({self.argument_module.name})"

    def forward(self, state):
        function = self.function_module.forward(state)
        argument = self.argument_module.forward(state)
        return torch.matmul(function, argument)

    def __str__(self):
        return self.name


# TODO: Turn this into a generative function
def spawn_extension_module(semantic_type):
    semantic_type_str = str(semantic_type)
    semantic_type_dims_dict = {"e": (32,), "<e,t>": (1, 32), "<e,<e,t>>": (1, 32, 32)}
    return ExtensionModule(output_dims=semantic_type_dims_dict[semantic_type_str])


def main():
    np = ExtensionModule(output_dims=(32,))
    snp = ExtensionModule(output_dims=(1, 32))
    snpnp = ExtensionModule(output_dims=(1, 32, 32))
    resize = transforms.Resize(64)
    image = read_image(f"taxi.png", torchvision.io.ImageReadMode.RGB)
    print(image.size())
    image = resize(image).type(torch.float)
    print(image.size())
    npe = np.forward(image.unsqueeze(0))
    snpe = snp.forward(image.unsqueeze(0))
    print(npe.size())
    print(snpe.size())
    print(torch.matmul(snpe, npe))
    snpnpe = snpnp.forward(image.unsqueeze(0))
    print(snpnpe.size())
    print(torch.matmul(torch.matmul(snpnpe, npe), npe))


def test():
    taxi = SemanticIntension(name="taxi", module=ExtensionModule(output_dims=(32,)))
    touch_n = SemanticIntension(name="touching_north", module=ExtensionModule(output_dims=(1, 32)))
    taxi_touching_n = SemanticIntensionApplication(function_module=touch_n, argument_module=taxi)

    resize = transforms.Resize(64)
    image = read_image(f"taxi.png", torchvision.io.ImageReadMode.RGB)
    image = resize(image).type(torch.float)

    print(taxi_touching_n.forward(image.unsqueeze(0)))
    print(taxi_touching_n)


if __name__ == "__main__":
    # main()
    test()
