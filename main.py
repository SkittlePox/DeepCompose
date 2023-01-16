from LexicalStructures import *
from Grammar import *
from LexiconParser import *


def taxi_example():
    lex_parser = LexiconParser()
    entries = lex_parser.parse_file("taxi_lexicon.txt")
    lexicon = Lexicon(list(set(entries)))
    print(lexicon)
    print(entries[2].semantics.semantic_type)
    # print(entries[2].semantics(entries[1].semantics))
    grammar = Grammar()
    interactor = GrammarInteractor(grammar)
    interactor.populate_lexicon(lexicon, layers=2)
    print("After populating:")
    print(lexicon)
    # tnp = lexicon.get_entry("touching_north(passenger)")
    # print(type(tnp.semantics))


if __name__ == "__main__":
    taxi_example()
