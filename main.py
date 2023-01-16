from LexicalStructures import *
from Grammar import *
from Trainer import *


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
    tnp = lexicon.get_entry("touching_north(passenger)")
    print(type(tnp.semantics))

    resize = transforms.Resize(64)
    image = read_image(f"taxi.png", torchvision.io.ImageReadMode.RGB)
    image = resize(image).type(torch.float)
    image = image.unsqueeze(0)

    print(tnp.semantics.forward(image))
    print(lexicon.get_entry("inside").semantics.forward(image))


def learning_propositions():
    lex_parser = LexiconParser()
    entries = lex_parser.parse_file("taxi_lexicon.txt")
    lexicon = Lexicon(list(set(entries)))
    grammar = Grammar()
    interactor = GrammarInteractor(grammar)
    interactor.populate_lexicon(lexicon, layers=2)

    # TERMS = ['touch_n(taxi, wall)', 'touch_s(taxi, wall)', 'touch_e(taxi, wall)',
    #          'touch_w(taxi, wall)', 'on(taxi, passenger)',
    #          'on(taxi, destination)', 'passenger.in_taxi']

    propositions = [lexicon.get_entry("touching_north(taxi)"),
                    lexicon.get_entry("touching_south(taxi)"),
                    lexicon.get_entry("touching_east(taxi)"),
                    lexicon.get_entry("touching_west(taxi)"),
                    lexicon.get_entry("on(passenger)(taxi)"),
                    lexicon.get_entry("on(destination)(taxi)"),
                    lexicon.get_entry("inside(taxi)(passenger)")]

    prop_module = PropositionSetModule(semantic_intensions=[prop.semantics for prop in propositions])

    file = open('../images/random_states.pkl', 'rb')
    dataset = PropositionDataset(root_dir="../images/random_states/", label_dict=pickle.load(file))


if __name__ == "__main__":
    learning_propositions()
