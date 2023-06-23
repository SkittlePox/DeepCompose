import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import torch.utils.data
import torch
import torchvision.transforms as transforms
from sklearn import decomposition

import deepcompose as dc

from utils import *
from tqdm import tqdm
import torch.nn as nn


def train(model, dataloader, num_epochs, plot=True, device="cuda"):
    """
    :param torch.nn.Module model: the model to be trained
    :param torch.utils.data.DataLoader dataloader: DataLoader containing training examples
    :param int num_epochs: number of epochs to train for
    :param torch.device device: the device that we'll be training on

    :return torch.nn.Module model
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    model.to(device)
    model.train()

    losses = []

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
        if device == "cuda":
            losses.append((average_loss / len(dataloader)).cpu().detach().numpy())
        else:
            losses.append((average_loss / len(dataloader)).detach().numpy())

    return model, losses


def evaluate(model, dataloader, device="cuda"):
    """ Evaluate a PyTorch Model

    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on

    :return accuracy
    """

    model.to(device)
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
            total_accuracy = (torch.sum(torch.sum((pred==y).float(), dim=0)) == 96).float()
        else:
            total_accuracy += (torch.sum(torch.sum((pred==y).float(), dim=0)) == 96).float()
        
        # print(total_accuracy.shape)

        progress_bar.update(1)

    print(total_accuracy)

    total_accuracy /= len(dataloader.dataset)
    print(total_accuracy)

    return total_accuracy


def taxi_example():
    lex_parser = dc.LexiconParser()
    entries = lex_parser.parse_file("taxi_lexicon.txt")
    lexicon = dc.Lexicon(list(set(entries)))
    print(lexicon)
    print(entries[2].semantics.semantic_type)
    # print(entries[2].semantics(entries[1].semantics))
    grammar = dc.Grammar()
    interactor = dc.ProductionGenerator(grammar)
    interactor.populate_lexicon(lexicon, layers=2)
    print("After populating:")
    print(lexicon)
    tnp = lexicon.get_entry("touching_north(passenger)")
    print(type(tnp.semantics))

    resize = transforms.Resize(64)
    image = read_image(f"taxi.png", torchvision.io.ImageReadMode.RGB)
    image = resize(image).type(torch.float)
    image = image.unsqueeze(0)

    new_image = transforms.ToPILImage()(image.squeeze())
    new_image = np.array(new_image)

    new_image = 255 - new_image

    plt.imshow(new_image)
    plt.show()

    print(tnp.semantics.forward(image))
    print(lexicon.get_entry("inside").semantics.forward(image))


def learning_propositions(epochs=20, batch_size=30, save=True, fixed_primitives=True):
    # Semantics.HIDDEN_DIM = 32
    lex_parser = dc.LexiconParser(fixed_primitives=fixed_primitives)
    entries = lex_parser.parse_file("taxi_lexicon.txt")
    lexicon = dc.Lexicon(list(set(entries)))
    grammar = dc.Grammar()
    interactor = dc.ProductionGenerator(grammar)
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

    prop_module = dc.PropositionSetModule(semantic_intensions=[prop.semantics for prop in propositions])

    # resize = transforms.Resize(64)
    # image = read_image(f"taxi.png", torchvision.io.ImageReadMode.RGB)
    # image = resize(image).type(torch.float)
    # image = image.repeat((10, 1, 1, 1))
    #
    # print(prop_module.forward(image))

    file = open('./images/random_states.pkl', 'rb')
    dataset = MDPPropositionDataset(root_dir="./images/random_states/", label_dict=pickle.load(file))
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=40, shuffle=False)

    # evaluate(prop_module, test_dataloader)
    _, losses = train(prop_module, train_dataloader, num_epochs=epochs)
    train_accuracy = evaluate(prop_module, train_dataloader)
    test_accuracy = evaluate(prop_module, test_dataloader)

    if save:
        torch.save(prop_module, 'propositions.pt')
        pickle.dump(lexicon, open('lexicon.pkl', 'wb'))

    return losses, test_accuracy, train_accuracy


def learning_propositions_extended(epochs=20, batch_size=30, save=True, fixed_primitives=True):
    # Semantics.HIDDEN_DIM = 32
    lex_parser = dc.LexiconParser(fixed_primitives=fixed_primitives)
    entries = lex_parser.parse_file("taxi_lexicon.txt")
    lexicon = dc.Lexicon(list(set(entries)))
    grammar = dc.Grammar()
    interactor = dc.ProductionGenerator(grammar)
    interactor.populate_lexicon(lexicon, layers=2)

    # TERMS = ['touch_n(taxi, wall)', 'touch_s(taxi, wall)', 'touch_e(taxi, wall)',
    #          'touch_w(taxi, wall)', 'on(taxi, passenger)',
    #          'on(taxi, destination)', 'passenger.in_taxi']
    # Additional terms: ['touch_n(passenger, wall)', 'touch_s(passenger, wall)', 'touch_e(passenger, wall)',
    #                    'touch_w(passenger, wall)', 'is(passenger, blue)', 'is(passenger, green)',
    #                    'is(passenger, red)', 'is(passenger, yellow)']

    print(lexicon)

    propositions = [lexicon.get_entry("touching_north(taxi)"),
                    lexicon.get_entry("touching_south(taxi)"),
                    lexicon.get_entry("touching_east(taxi)"),
                    lexicon.get_entry("touching_west(taxi)"),
                    lexicon.get_entry("on(passenger)(taxi)"),
                    lexicon.get_entry("on(destination)(taxi)"),
                    lexicon.get_entry("inside(taxi)(passenger)"),
                    lexicon.get_entry("touching_north(passenger)"),
                    lexicon.get_entry("touching_south(passenger)"),
                    lexicon.get_entry("touching_east(passenger)"),
                    lexicon.get_entry("touching_west(passenger)"),
                    ]


def param_sweep(fixed_primitives=False, epochs=30):
    hidden_dim_params = [2, 4, 8, 16, 32, 64, 128, 256][:]

    for hd in hidden_dim_params:
        dc.Semantics.HIDDEN_DIM = hd
        losses, test_accuracy, train_accuracy = learning_propositions(epochs=epochs, save=False, fixed_primitives=fixed_primitives)
        plt.plot(losses, label=f"hd={hd} test_acc={str(torch.mean(test_accuracy).tolist())[:5]}"
                               f" train_acc={str(torch.mean(train_accuracy).tolist())[:5]}")

    # if fixed_primitives:
    #     Semantics.HIDDEN_DIM = 256
    #     losses, test_accuracy, train_accuracy = learning_propositions(epochs=epochs, save=False,
    #                                                                   fixed_primitives=False)
    #     plt.plot(losses, label=f"unfixed hd={256} test_acc={str(torch.mean(test_accuracy).tolist())[:5]}"
    #                            f" train_acc={str(torch.mean(train_accuracy).tolist())[:5]}")

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()


def probe():
    # model = torch.load("propositions.pt")
    # print(type(model))
    # print(model.semantic_intensions)

    lexicon = pickle.load(open("lexicon.pkl", 'rb'))

    file = open('./images/random_states.pkl', 'rb')
    dataset = MDPPropositionDataset(root_dir="./images/random_states/", label_dict=pickle.load(file))
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)

    # entry_names = [("taxi", 0), ("passenger", 1), ("destination", 2)]
    entry_names = [("touching_north", 0), ("touching_south", 1), ("touching_east", 2), ("touching_west", 3)][:1]
    # entry_names = [("on(passenger)", 0), ("on(destination)", 1), ("inside(taxi)", 2)]

    entry_models = [lexicon.get_entry(e_name).semantics for e_name, _ in entry_names]
    [e_model.eval() for e_model in entry_models]

    preds = []
    labels = []
    images = []

    for i, batch in enumerate(dataloader):
        X, y = batch[0], batch[1]
        with torch.no_grad():
            for i, e_model in enumerate(entry_models):
                images.extend(X)
                preds.append(e_model(X).squeeze())
                labels.append(torch.ones((200,)) * i)

    X = torch.cat(preds)
    y = torch.cat(labels)

    print(len(images))
    print(X.size())
    print(y.size())

    pca = decomposition.PCA(n_components=3)

    pca.fit(X)
    X = pca.transform(X)

    fig = plt.figure(2, figsize=(4, 3))
    plt.clf()

    ax = fig.add_subplot(121, projection="3d", elev=48, azim=134)
    # ax.set_position([0, 0, 0.95, 1])

    for name, label in entry_names:
        ax.text3D(
            X[y == label, 0].mean(),
            X[y == label, 1].mean(),
            X[y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

    cursor = mplcursors.cursor(ax, hover=True)

    ax2 = fig.add_subplot(122)
    image = images[0].type(torch.uint8)
    new_image = transforms.ToPILImage()(image)
    im = ax2.imshow(new_image)

    @cursor.connect("add")
    def on_add(sel):
        # print(sel.index)
        image = images[sel.index].type(torch.uint8)
        image = transforms.ToPILImage()(image)
        # ax2.imshow(image)
        im.set_data(image)

    plt.show()


def propositional_logic_experiment_clevr(epochs=1, batch_size=128, save=False):
    # model = dc.PropositionalPrimitive()
    model = torch.load("saved_models/proposition_primitives_0.pt")
    # model.to('cpu')

    trainA = CLEVR96ClassifierDataset(scene_file="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/scenes/clevr_ref+_cogent_trainA_scenes.json",
                                    images_dir="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/images/trainA/",
                                    label_file="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/labels/clevr_ref+_cogent_trainA_labels.json")
    valA = CLEVR96ClassifierDataset(scene_file="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/scenes/clevr_ref+_cogent_valA_scenes.json",
                                    images_dir="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/images/valA/",
                                    label_file="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/labels/clevr_ref+_cogent_valA_labels.json")
    valB = CLEVR96ClassifierDataset(scene_file="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/scenes/clevr_ref+_cogent_valB_scenes.json",
                                    images_dir="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/images/valB/",
                                    label_file="/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/labels/clevr_ref+_cogent_valB_labels.json")
    
    trainA_loader = DataLoader(trainA, batch_size=batch_size, shuffle=True)
    valA_loader = DataLoader(valA, batch_size=batch_size, shuffle=True)
    valB_loader = DataLoader(valB, batch_size=batch_size, shuffle=True)

    # _, losses = train(model, trainA_loader, num_epochs=epochs)
    trainA_accuracy = evaluate(model, trainA_loader)
    valA_accuracy = evaluate(model, valA_loader)
    valB_accuracy = evaluate(model, valB_loader)

    print(f"trainA accuracy: {trainA_accuracy}")
    print(f"valA accuracy: {valA_accuracy}")
    print(f"valB accuracy: {valB_accuracy}")

    model.to('cpu')

    if save:
        torch.save(model, 'saved_models/proposition_primitives_0.pt')


def propositional_logic_experiment_emnist(epochs=1, batch_size=64, save=False):
    # Create a PropositionPrimitive for each digit 0-4
    
    primitives = []
    for i in range(5):
        primitives.append(dc.PropositionalPrimitive(i))
    
    model = dc.PropositionPrimitiveCollection(primitives)



if __name__ == "__main__":
    # param_sweep(fixed_primitives=True, epochs=20)
    # learning_propositions(epochs=10, save=False)
    # learning_propositions_extended(epochs=10, save=False)
    # probe()
    # taxi_example()
    propositional_logic_experiment_emnist(batch_size=64, save=False)
