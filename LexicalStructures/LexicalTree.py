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
