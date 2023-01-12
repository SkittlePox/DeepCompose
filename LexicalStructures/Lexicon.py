from LexicalStructures.LexicalEntry import *


class Lexicon:
    def __init__(self, entries=[]):
        self.entries = list(set(entries))  # A list of LexicalEntrys
        self.new_id = 1
        self.entries.sort(key=lambda x: x.semantics.complexity())  # Sorts by complexity

        for e in self.entries:
            e.id = self.new_id
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

    def getEntry(self, id_or_english):
        entry = None
        for e in self.entries:
            if e.id == id_or_english or e.english == id_or_english:
                return e

    def __str__(self):
        out = "=== Lexicon ===\n"
        for e in self.entries:
            out = out + str(e) + "\n"
        return out + "==============="
