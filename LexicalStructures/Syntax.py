import enum
from functools import reduce

VERBOSE = True


class SyntacticPrimitive(enum.Enum):
    def __str__(self):
        return str(self.value)

    S = "S"
    NP = "NP"
    PP = "PP"
    N = "N"
    CP = "CP"


class SyntacticFeature(enum.Enum):
    def __str__(self):
        return self.value

    A = "A"
    V = "V"
    TO = "TO"
    OF = "OF"
    GEN = "GEN"


class SyntacticSlash(enum.Enum):
    def __str__(self):
        if VERBOSE:
            return self.value
        else:
            return ""

    L = "˻"
    R = "ʳ"


class SyntacticCategory:
    def __init__(self, lhs=None, rhs=None, slash=None, features=None):
        self.lhs = lhs
        self.rhs = rhs  # This is possibly None
        self.features = features
        self.slash = slash

        if slash == None and self.rhs is not None:  # Word order rules
            if self.lhs.lhs == SyntacticPrimitive.S or self.lhs == self.rhs:
                self.slash = SyntacticSlash.L
            else:
                self.slash = SyntacticSlash.R

    def optional_features(self):
        if self.features is None:
            return ""
        else:
            return "[" + reduce(lambda a, b: f"{a}][{b}", self.features) + "]"

    def possible_primitive(self):
        if self.rhs is None:
            return f"{str(self.lhs)}"
        else:
            return f"({str(self.lhs)}/{str(self.slash)}{str(self.rhs)})"

    def __str__(self):
        return f"{self.possible_primitive()}{self.optional_features()}"

    def __eq__(self, other):
        return isinstance(other, SyntacticCategory) \
               and self.lhs == other.lhs and self.rhs == other.rhs \
               and self.features == other.features and self.slash == other.slash
