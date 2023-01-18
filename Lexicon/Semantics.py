import enum
import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torchvision import transforms
from functools import reduce


class SemanticTypePrimitive(enum.Enum):
    def __str__(self):
        return str(self.value)

    e = "e"  # Individual
    t = "t"  # Truth value


class SemanticType:
    """
    This object contains a semantic type like <e,t>
    A linguistic expression with type <e,t> is a function from individuals (e) to truth values (t)
    """

    def __init__(self, lhs=None, rhs=None):
        self.lhs = lhs
        self.rhs = rhs

    def complexity(self):
        """
        This is for sorting purposes
        """
        if self.lhs is None:
            return 1
        lhs = 1 if isinstance(self.lhs, SemanticTypePrimitive) else self.lhs.complexity()
        rhs = 1 if isinstance(self.rhs, SemanticTypePrimitive) else self.rhs.complexity()
        return lhs + rhs

    def __call__(self, argument):
        return self.rhs

    def __str__(self):
        if self.lhs is None:
            return str(self.rhs)
        else:
            return f"<{str(self.lhs)},{str(self.rhs)}>"

    def __eq__(self, other):
        return isinstance(other, SemanticType) \
               and self.lhs == other.lhs and self.rhs == other.rhs


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        # print(x.size())
        # print(self.dims)
        out = x.view(*self.dims)
        # print(out.size())
        return out


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
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.Tanh(),
            Flatten(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, abs(reduce(lambda x, y: x * y, output_dims))),
            UnFlatten(dims=output_dims)
        )

    def forward(self, state):
        return self.intension(state)


class SemanticEntry(nn.Module):
    def __init__(self, name, semantic_type):
        super().__init__()
        self.name = name
        self.semantic_type = semantic_type

    def complexity(self):
        return self.semantic_type.complexity()

    def forward(self, state):
        return None

    def compose(self, argument):
        return SemanticIntensionApplication(function_module=self, argument_module=argument)

    def __str__(self):
        return f"Intension[{self.name}]"


class SemanticIntensionPrimitive(SemanticEntry):
    def __init__(self, name, module, semantic_type):
        super().__init__(name=name, semantic_type=semantic_type)
        self.module = module
        self.add_module(name, module)

    def forward(self, state):
        return self.module.forward(state)


class SemanticIntensionApplication(SemanticEntry):
    def __init__(self, function_module, argument_module):
        """
        function_module: SemanticEntry
        argument_module: SemanticEntry
        """
        super().__init__(name=f"{function_module.name}({argument_module.name})",
                         semantic_type=function_module.semantic_type(argument_module.semantic_type))
        self.function_module = function_module
        self.argument_module = argument_module
        self.add_module(function_module.name, function_module)
        self.add_module(argument_module.name, argument_module)
        self.unflatten = UnFlatten(dims=get_semantic_type_dims(self.function_module.semantic_type.rhs))

    def forward(self, state):
        function = self.function_module.forward(state)
        argument = self.argument_module.forward(state)
        # print(function.size())
        # print(argument.size())
        # print(self.function_module.semantic_type)

        comp = torch.matmul(function, argument)
        comp = self.unflatten(comp)
        return torch.sigmoid(comp)


class PropositionSetModule(nn.Module):
    def __init__(self, semantic_intensions):
        super().__init__()
        self.semantic_intensions = semantic_intensions
        [self.add_module(prop.name, prop) for prop in semantic_intensions]

    def forward(self, state):
        outputs = [prop.forward(state) for prop in self.semantic_intensions]
        return torch.cat(outputs, dim=1)


# TODO: Turn this into a generative function
def get_semantic_type_dims(semantic_type, hidden_dim=64):
    semantic_type_str = str(semantic_type)
    semantic_type_dims_dict = {"e": (-1, hidden_dim, 1), "t": (-1, 1), "<e,t>": (-1, 1, hidden_dim),
                               "<e,<e,t>>": (-1, hidden_dim, hidden_dim)}
    return semantic_type_dims_dict[semantic_type_str]


def main():
    np = ExtensionModule(output_dims=(-1, 32, 1))
    snp = ExtensionModule(output_dims=(-1, 1, 32))
    snpnp = ExtensionModule(output_dims=(-1, 32, 32))
    resize = transforms.Resize(64)
    image = read_image(f"taxi.png", torchvision.io.ImageReadMode.RGB)
    # print(image.size())
    image = resize(image).type(torch.float)
    image = image.repeat((10, 1, 1, 1))
    # print(image.size())
    npe = np.forward(image)
    snpe = snp.forward(image)
    snpnpe = snpnp.forward(image)

    print(f"npe size: {npe.size()}")
    print(f"snpe size: {snpe.size()}")
    print(f"snpnpe size: {snpnpe.size()}")

    s = torch.matmul(snpe, npe)
    print(f"snpe(npe) = s size: {s.size()}")
    unflatten = UnFlatten(dims=(-1, 1, 1))
    s = unflatten.forward(s)
    print(f"snpe(npe) = s size: {s.size()} (unflattened)")

    new_snpe = torch.matmul(snpnpe, npe)
    print(f"snpnpe(npe) = snp size: {new_snpe.size()}")
    unflatten = UnFlatten(dims=(-1, 1, 32))
    new_snpe = unflatten.forward(new_snpe)
    print(f"snpnpe(npe) = snp size: {new_snpe.size()} (unflattened)")

    new_s = torch.matmul(new_snpe, npe)
    print(f"snpnpe(npe)(npe) = s size: {new_s.size()}")

    # print(torch.matmul(snpe, npe))
    # snpnpe = snpnp.forward(image)
    # print(f"snpnpe size: {snpnpe.size()}")
    # print(npe.size())

    # torch.matmul(new_snpe, npe)
    # print(torch.matmul(torch.matmul(snpnpe, npe), npe))


def test():
    type_e = SemanticTypePrimitive.e
    type_et = SemanticType(lhs=type_e, rhs=SemanticTypePrimitive.t)
    taxi = SemanticIntensionPrimitive(name="taxi", module=ExtensionModule(output_dims=(-1, 32, 1)),
                                      semantic_type=type_e)
    touch_n = SemanticIntensionPrimitive(name="touching_north", module=ExtensionModule(output_dims=(-1, 1, 32)),
                                         semantic_type=type_et)
    taxi_touching_n = SemanticIntensionApplication(function_module=touch_n, argument_module=taxi)

    resize = transforms.Resize(64)
    image = read_image(f"taxi.png", torchvision.io.ImageReadMode.RGB)
    image = resize(image).type(torch.float)
    image = image.repeat((2, 1, 1, 1))
    print(image.size())

    print(taxi_touching_n.forward(image))
    print(taxi_touching_n)


if __name__ == "__main__":
    # main()
    test()
