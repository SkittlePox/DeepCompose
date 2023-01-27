import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from utils import *

TERMS = ['touch_n(taxi, wall)', 'touch_s(taxi, wall)', 'touch_e(taxi, wall)',
         'touch_w(taxi, wall)', 'on(taxi, passenger)',
         'on(taxi, destination)', 'passenger.in_taxi']


def proposition_distribution():
    file = open('./images/random_states.pkl', 'rb')
    dataset = PropositionDataset(root_dir="./images/random_states/", label_dict=pickle.load(file))
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)

    y = None

    for i, batch in enumerate(dataloader):
        _, y = batch[0], batch[1]

    y = torch.sum(y, dim=0).numpy() / 200.0

    fig, ax = plt.subplots()
    ax.bar(TERMS, y)
    ax.tick_params(axis='x', labelrotation=45)
    ax.axhline(y=0.5, color='r', linestyle='-')

    ax.set_ylabel('Proportion of states when true')
    ax.set_title('Distribution of Propositions over Taxi States')

    plt.show()


def state_distribution():
    file = open('./images/random_states.pkl', 'rb')
    dataset = PropositionDataset(root_dir="./images/random_states/", label_dict=pickle.load(file))
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)

    y = None

    for i, batch in enumerate(dataloader):
        _, y = batch[0], batch[1]

    y = torch.sum(y, dim=1).numpy()

    ax = sns.histplot(y, discrete=True)
    ax.set_ylabel("Number of States")
    ax.set_xlabel("Number of true propositions")
    ax.set_title("Number of True Propositions per State")
    plt.show()


def proposition_cooccurence():
    file = open('./images/random_states.pkl', 'rb')
    dataset = PropositionDataset(root_dir="./images/random_states/", label_dict=pickle.load(file))
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)

    y = None

    for i, batch in enumerate(dataloader):
        _, y = batch[0], batch[1]

    y = y.numpy()

    cooccurrence_matrix = np.dot(y.transpose(), y)

    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(
            np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))

    covariance_matrix = np.cov(y, rowvar=False)
    corrcoef_matrix = np.corrcoef(y, rowvar=False)

    ax = sns.heatmap(corrcoef_matrix, linewidth=0.5, annot=True, xticklabels=TERMS, yticklabels=TERMS, cmap="crest")
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='left')
    ax.set_title("Correlation Co-efficients")
    plt.show()


if __name__ == "__main__":
    # proposition_distribution()
    proposition_cooccurence()
    # state_distribution()
