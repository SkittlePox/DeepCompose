# Direct-Compositional-Parser

This is an implementation of the Direct Compositional semantic framework as given in CLPS 1342: Compositional Semantics

Add entries to lexicon.txt using the following pattern and run `main.py`. Parsing rules are *not* enforced.

```
Zinkly ; NP ; e ; z
walks ; S/NP ; <e,t> ; m=1 p=0 z=0
< "Zinkly walks" ; S ; t ; walks'(z) ; 0 >
```

 This framework has support for custom grammar rules, which require a `test()` and `operate()` function:

```python
 class CombinatoryRule():   
  	def __init__(self, name, test, operate):
        """
        test takes whole lexical entries and only looks at their syntax
        Returns whether or not this rule can operate on the given syntax categories

        operate takes whole lexical entries
        Returns the resulting lexical entry after applying this rule
        """
        self.name = name
        self.test = test
        self.operate = operate

    def __call__(self, a, b):
        return self.operate(a, b)

    def __str__(self):
        return self.name
```



Example rule:

```python
### R-1a

def R1a_test(a, b):
    return a.syntax.slash == SyntacticSlash.R and a.syntax.rhs == b.syntax

def R1a_operate(a, b):
    new_english = f"{a.english} {b.english}"
    new_category = a.category.lhs
    new_semantics = a.semantics(b.semantics)
    return LexicalEntry(english=new_english, syntacticCategory=new_category, semanticEntry=new_semantics)
```



## A More Thorough Explanation

All linguistic expressions are triples: `<[sound], Syntax Category, [[semantic meaning]]>`

Expressions are combined and lifted using combinatory and unary rules. Combinatory rules take two expressions and combine to form a new expression, unary rules apply to a single expression and form a new expresion. These rules compose a grammar.

The semantic meaning of a linguistic expression is represented by a lambda calculus expression like `likes(Porky)(Mitka)`, "Mitka likes treats." Here's what the entry for `likes` might look like:

```
likes ; (S/NP)/NP ; <e,<e,t>> ; m=(m=0 p=0) p=(m=1 p=0)
```



### The Code

The `LexicalStructures` package is one of the most significant packages, definitely take a look. The two most important files in this package are `LexicalEntry.py` and `Semantics.py`.

The `Grammar` package is also significant, it houses the grammar rules and the `Grammar` object.

A `LexicalEntry` is a representation of a linguistic expression. `Lexicon` objects are collections of `LexicalEntry`s. Grammar rules operate on one or multiple `LexicalEntry`s.

`Semantics.py` contains all of the semantic structures. There is a primary semantic object that contains substructures, this is the `SemanticEntry`. A `SemanticEntry` has a semantic type, extension and intention. The extension is a `LambdaCalcExpression`, this is a critical object that does the actual semantic calculation. Right now it's a glorified dictionary lookup, but my goal is to change this to a more sophisticated structure.
