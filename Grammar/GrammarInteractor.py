from itertools import compress
import copy
from Grammar import *


class GrammarInteractor():
    def __init__(self, grammar):
        self.grammar = grammar

    def possible_unaries(self, a):
        unary_rules = list(filter(lambda x: isinstance(x, UnaryRule), self.grammar.rules.values()))
        applicable_rules = []
        for r in unary_rules:
            if r.test(a):
                applicable_rules.append(r)
        return applicable_rules

    def possible_combinations(self, a, b):
        combinatory_rules = list(filter(lambda x: isinstance(x, CombinatoryRule), self.grammar.rules.values()))
        applicable_rules = []
        for r in combinatory_rules:
            if r.test(a, b):
                applicable_rules.append(r)
        return applicable_rules

    def populate_lexicon(self, lexicon, layers=4):
        for i in range(layers):
            entries = copy.copy(lexicon.entries)
            for a in entries:
                unary_rules = self.possible_unaries(a)
                if len(unary_rules) != 0:
                    un_entries = list(map(lambda x: x(a), unary_rules))
                    print(un_entries[0])
                    lexicon.add(un_entries)
                for b in entries:
                    combinatory_rules = self.possible_combinations(a, b)
                    if len(combinatory_rules) != 0:
                        new_entries = list(map(lambda x: x(a, b), combinatory_rules))
                        lexicon.add(new_entries)
