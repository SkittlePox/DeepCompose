from LexicalStructures import *
from LexiconParser import *

parser = LexiconParser()


### R-1a

def R1a_test(a, b):
    return a.syntax.slash == SyntacticSlash.R and a.syntax.rhs == b.syntax


def R1a_operate(a, b):
    new_english = f"{a.english} {b.english}"
    new_category = a.syntax.lhs
    new_semantics = a.semantics(b.semantics)
    return LexicalEntry(english=new_english, syntacticEntry=new_category, semanticEntry=new_semantics)


### R-1b

def R1b_test(a, b):
    return b.syntax.slash == SyntacticSlash.L and b.syntax.rhs == a.syntax


def R1b_operate(a, b):
    new_english = f"{a.english} {b.english}"
    new_category = b.syntax.lhs
    new_semantics = b.semantics(a.semantics)
    return LexicalEntry(english=new_english, syntacticEntry=new_category, semanticEntry=new_semantics)


### R-2         This is a unary rule!

AP = parser.parse_syntactic_category("S[A]/rNP")


def R2_test(a):
    # print(a.syntax == AP)
    return a.syntax == AP


def R2_operate(a):
    new_category = parser.parse_syntactic_category("N/N") if " " in a.english else parser.parse_syntactic_category(
        "N/rN")

    def given_P(P):
        def given_x(x):
            return a.semantics.extension.function(x) and P(x)

        return given_x

    extension = LambdaCalcExpression(given_P)
    intention = SemanticIntention(argument=f"{str(a.semantics.intention)} - nmod")
    type = parser.parse_semantic_type("<e,<e,<e,t>>>")
    new_semantics = SemanticEntry(intention=intention, extension=extension, type=type)
    return LexicalEntry(english=a.english, syntacticEntry=new_category, semanticEntry=new_semantics)
