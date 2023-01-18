SPACING = True


class Lexicon:
    def __init__(self, entries=[]):
        self.entries = list(set(entries))  # A list of LexicalEntrys
        self.new_id = 1
        self.entries.sort(key=lambda x: len(str(x.syntax)))  # Sorts by complexity

        for e in self.entries:
            e.semantic_id = self.new_id
            self.new_id += 1

    def insort(self, entry):
        """
        I should be using bisect.insort with a key, but the developers who wrote
        the bisect library thought it would cause developers to write inefficient code
        well, take this, O(n)
        """
        entry_complexity = entry.semantics.complexity()
        for i in range(len(self.entries)):
            if self.entries[i].semantics.complexity() > entry_complexity:
                self.entries.insert(i, entry)
                return
        self.entries.insert(-1, entry)

    def add(self, entry):
        if isinstance(entry, LexicalEntry):
            entry = [entry]
        for e in entry:
            if not self.contains(e):
                e.id = self.new_id
                self.new_id += 1
                self.insort(e)

    def contains(self, entry):
        for e in self.entries:
            if entry == e:
                return True
        return False

    def get_entry(self, id_or_english):
        for e in self.entries:
            if e.semantics.name == id_or_english or e.english == id_or_english:
                return e
        return None

    def __str__(self):
        out = "=== Lexicon ===\n"
        for e in self.entries:
            out = out + str(e) + "\n"
        return out + "==============="


class LexicalTree:
    """
    A LexicalTree stores two LexicalEntrys and a single CombinatoryRule
    """

    def __init__(self, rule, a, b):
        self.rule = rule
        self.a = a
        self.b = b

    def evaluate(self):
        a = self.a.evaluate() if isinstance(self.a, LexicalTree) else self.a
        b = self.b.evaluate() if isinstance(self.b, LexicalTree) else self.b
        return self.rule(a, b)

    def __str__(self):  # Replace this with a legit tree printing alg
        eng_a = str(self.a) if isinstance(self.a, LexicalTree) else self.a.english
        eng_b = str(self.b) if isinstance(self.b, LexicalTree) else self.b.english
        return f"(\'{self.rule.name}\' => {eng_a} + {eng_b})"


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
