from itertools import compress
import copy
from Grammar import *


class GrammarInteractor():
    def __init__(self, grammar):
        self.grammar = grammar

    def possible_unaries(self, a):
        un_rules = list(filter(lambda x: isinstance(x, UnaryRule), self.grammar.rules.values()))
        rls = []
        for r in un_rules:
            if r.test(a):
                rls.append(r)
        return rls

    def possible_combinations(self, a, b):
        co_rules = list(filter(lambda x: isinstance(x, CombinatoryRule), self.grammar.rules.values()))
        rls = []
        for r in co_rules:
            if r.test(a, b):
                rls.append(r)
        return rls

    def populate_lexicon(self, lexicon, layers=4):
        for i in range(layers):
            entries = copy.copy(lexicon.entries)
            for a in entries:
                unaryRules = self.possible_unaries(a)
                if len(unaryRules) != 0:
                    un_entries = list(map(lambda x: x(a), unaryRules))
                    print(un_entries[0])
                    lexicon.add(un_entries)
                for b in entries:
                    combinatoryRules = self.possible_combinations(a, b)
                    if len(combinatoryRules) != 0:
                        newEntries = list(map(lambda x: x(a, b), combinatoryRules))
                        lexicon.add(newEntries)
