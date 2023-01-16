import enum
from .TorchSemantics import *


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


class SemanticEntry:
    def __init__(self, semantic_id, intension=None, semantic_type=None):
        """
        Takes a SemanticIntension and SemanticType
        """
        self.semantic_id = semantic_id
        self.intension = intension
        self.semantic_type = semantic_type

    def complexity(self):
        return self.semantic_type.complexity()

    def calculate_extension(self, state):
        return self.intension.forward(state)

    def __call__(self, argument):
        """
        argument is another SemanticEntry
        """
        intension = SemanticIntensionApplication(function_module=self.intension, argument_module=argument.intension)
        semantic_type = self.semantic_type(argument.semantic_type)
        return SemanticEntry(semantic_id=f"{self.semantic_id}({argument.semantic_id})", intension=intension, semantic_type=semantic_type)

    def __str__(self):
        baseStr = ""
        baseStr += str(self.semantic_type)
        # baseStr += " ; "
        # baseStr += str(self.id)
        baseStr += " ; "
        baseStr += str(self.intension)
        return baseStr
