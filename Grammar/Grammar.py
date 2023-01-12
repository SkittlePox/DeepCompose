from .GrammarRules import *


class Rule():
    def __init__(self, name, test, operate, description=""):
        """
        test takes whole lexical entries and only looks at their syntax
        Returns whether or not this rule can operate on the given syntax categories

        operate takes whole lexical entries
        Returns the resulting lexical entry after applying this rule
        """
        self.name = name
        self.test = test
        self.operate = operate
        self.description = description

    def __call__(self, a, b):
        return self.operate(a, b)

    def __str__(self):
        return f"{self.name}\n{self.description}"


class CombinatoryRule(Rule):
    pass


class UnaryRule(Rule):
    def __call__(self, a):
        return self.operate(a)


class Grammar():
    def __init__(self):
        rules = {}
        rules['R-1a'] = (
            CombinatoryRule(name="R-1a", test=R1a_test, operate=R1a_operate, description="Some description here"))
        rules['R-1b'] = (CombinatoryRule(name="R-1b", test=R1b_test, operate=R1b_operate))
        rules['R-2'] = (UnaryRule(name="R-2", test=R2_test, operate=R2_operate))
        self.rules = rules
