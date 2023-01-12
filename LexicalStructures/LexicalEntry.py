import enum
from functools import reduce

SPACING = True


class LexicalEntry:
    """
    A LexicalEntry is a representation of a linguistic expression which is a triplet:
    <[sound], syntax category, [[semantic meaning]]>
    """

    def __init__(self, english, syntacticEntry, semanticEntry, id=-1):
        """
        english is a string that represents the [sound] of the expression, like 'walks'
        syntacticEntry is a SyantacticCategory from Syntax.py
        semanticEntry is a SemanticEntry from Semantics.py
        """
        self.english = english
        self.syntax = syntacticEntry
        self.semantics = semanticEntry
        self.id = id

    def __eq__(self, other):
        return isinstance(other, LexicalEntry) \
               and self.english == other.english \
               and self.syntax == other.syntax
        # and self.semantics.type == other.semantics.type #and self.semantics.function == other.semantics.function

    def __str__(self):
        syntax = str(self.syntax)
        semantics = str(self.semantics)
        if SPACING:
            return f"<[{self.id}] \"{self.english}\" ; {syntax} ; {semantics} >"
        else:
            return f"<\"{self.english}\";{syntax};{semantics}>"

    def __hash__(self):
        return hash(self.english)  # LOL easy way out
