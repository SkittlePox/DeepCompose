from Lexicon import *


### R-1a
from Lexicon import LexicalEntry


def R1a_test(a, b):
    return a.syntax.slash == SyntacticSlash.R and a.syntax.rhs == b.syntax


def R1a_operate(a, b):
    new_english = f"{a.english} {b.english}"
    new_category = a.syntax.lhs
    new_semantics = a.semantics.compose(b.semantics)
    return LexicalEntry(english=new_english, syntax_entry=new_category, semantic_entry=new_semantics)


### R-1b

def R1b_test(a, b):
    return b.syntax.slash == SyntacticSlash.L and b.syntax.rhs == a.syntax


def R1b_operate(a, b):
    new_english = f"{a.english} {b.english}"
    new_category = b.syntax.lhs
    new_semantics = b.semantics.compose(a.semantics)
    return LexicalEntry(english=new_english, syntax_entry=new_category, semantic_entry=new_semantics)


### R-2         This is a unary rule!

# AP = parser.parse_syntactic_category("S[A]/rNP")


# def R2_test(a):
#     # print(a.syntax == AP)
#     return a.syntax == AP
#
#
# def R2_operate(a):
#     new_category = parser.parse_syntactic_category("N/N") if " " in a.english else parser.parse_syntactic_category(
#         "N/rN")
#
#     def given_P(P):
#         def given_x(x):
#             return a.semantic_id.extension.function(x) and P(x)
#
#         return given_x
#
#     extension = LambdaCalcExpression(given_P)
#     intention = SemanticIntension(argument=f"{str(a.semantics.intension)} - nmod")
#     type = parser.parse_semantic_type("<e,<e,<e,t>>>")
#     new_semantics = SemanticEntry(intension=intention, extension=extension, semantic_type=type)
#     return LexicalEntry(english=a.english, syntax_entry=new_category, semanticEntry=new_semantics)
