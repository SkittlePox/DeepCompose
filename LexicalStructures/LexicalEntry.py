import enum
from functools import reduce

SPACING = True


class LexicalEntry:
    """
    A LexicalEntry is a representation of a linguistic expression which is a triplet:
    <[sound], syntax category, [[semantic meaning]]>
    """

    def __init__(self, english, syntax_entry, semantic_entry, id=-1):
        """
        english is a string that represents the [sound] of the expression, like 'walks'
        syntax_entry is a SyantacticCategory from Syntax.py
        semantic_entry is a SemanticEntry from Semantics.py
        """
        self.english = english
        self.syntax = syntax_entry
        self.semantics = semantic_entry
        self.id = id

    def __eq__(self, other):
        return isinstance(other, LexicalEntry) \
               and self.english == other.english \
               and self.syntax == other.syntax \
               and self.semantics == other.semantics

    def __str__(self):
        syntax = str(self.syntax)
        semantics = str(self.semantics)
        if SPACING:
            return f"<[{self.id}] \"{self.english}\" ; {syntax} ; {semantics} >"
        else:
            return f"<\"{self.english}\";{syntax};{semantics}>"

    def __hash__(self):
        return hash(self.english + str(self.semantics))
