from LexicalStructures import *
from Grammar import *
from Trainer import *
from tqdm import tqdm
import torch.nn as nn


def train(model, dataloader, num_epochs, device="cpu"):
    """
    :param torch.nn.Module model: the model to be trained
    :param torch.utils.data.DataLoader dataloader: DataLoader containing training examples
    :param int num_epochs: number of epochs to train for
    :param torch.device device: the device that we'll be training on

    :return torch.nn.Module model
    """

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.95)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} training:")
        progress_bar = tqdm(range(len(dataloader)))
        average_loss = 0.0

        for i, batch in enumerate(dataloader):
            X, y = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            pred = model(X)
            # print(pred.size())
            # print(y.size())

            # loss = nn.functional.nll_loss(pred.transpose(1, 2), y, ignore_index=QUERY_PAD_INDEX)
            loss = mse_loss(pred, y)
            progress_bar.update(1)
            average_loss += loss

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # print(f"Batch loss: {loss}")

        print(f"Average loss: {average_loss / len(dataloader)}")

    return model


def evaluate(model, dataloader, device="cpu"):
    """ Evaluate a PyTorch Model

    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on

    :return accuracy
    """

    model.eval()

    print("Evaluating:")

    progress_bar = tqdm(range(len(dataloader)))

    total_accuracy = None

    for i, batch in enumerate(dataloader):
        X, y = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            pred = model(X)
            pred = (pred > 0.5).float()

        if total_accuracy is None:
            total_accuracy = torch.sum((pred==y).float(), dim=0)
        else:
            total_accuracy += torch.sum((pred==y).float(), dim=0)

        progress_bar.update(1)

    total_accuracy /= len(dataloader.dataset)
    print(total_accuracy)

    return total_accuracy


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

    # resize = transforms.Resize(64)
    # image = read_image(f"taxi.png", torchvision.io.ImageReadMode.RGB)
    # image = resize(image).type(torch.float)
    # image = image.repeat((10, 1, 1, 1))
    #
    # print(prop_module.forward(image))

    file = open('./images/random_states.pkl', 'rb')
    dataset = PropositionDataset(root_dir="./images/random_states/", label_dict=pickle.load(file))
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    evaluate(prop_module, dataloader)
    train(prop_module, dataloader, num_epochs=15)
    evaluate(prop_module, dataloader)


if __name__ == "__main__":
    learning_propositions()
