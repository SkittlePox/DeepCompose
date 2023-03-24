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

    ax.set_ylabel('ProporDistribution of Propositions over Taxi Statetion of states when true')
    ax.set_title('s')

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


def rollout_proposition_changes(dir_name, num_agents=5):
    all_oo_states = []
    for a in range(num_agents):
        file = open(f'./images/{dir_name}/agent-{a}-info.pkl', 'rb')
        info = pickle.load(file)
        all_oo_states.append(info['oo_states'])

    def calc_prop_changes_aggregate(oo_states):
        prop_changes = []
        for i in range(len(oo_states)-1):
            prop_changes.append(np.sum(np.bitwise_xor(oo_states[i], oo_states[i+1])))
        return prop_changes

    def calc_prop_changes_piecemeal(oo_states):
        prop_changes = []
        for i in range(len(oo_states)-1):
            prop_changes.append(np.bitwise_xor(oo_states[i], oo_states[i+1]))
        return prop_changes

    all_prop_changes = [calc_prop_changes_aggregate(o_s) for o_s in all_oo_states]
    mean_prop_changes = [np.mean(p_c) for p_c in all_prop_changes]
    print(f"Mean Prop changes per step for {dir_name}: {np.mean(mean_prop_changes)}")

    piece_prop_changes = [np.stack(calc_prop_changes_piecemeal(o_s)) for o_s in all_oo_states]
    mean_piece_prop_changes = [np.mean(p_c, axis=0) for p_c in piece_prop_changes]
    mean_mean_piece_prop_changes = np.mean(np.stack(mean_piece_prop_changes), axis=0)
    print(f"Mean Prop changes per step (piecemeal) for {dir_name}: {mean_mean_piece_prop_changes}")

    fig, ax = plt.subplots()
    ax.bar(TERMS, mean_mean_piece_prop_changes)
    ax.tick_params(axis='x', labelrotation=45)
    ax.axhline(y=0.5, color='r', linestyle='-')

    ax.set_ylabel('Chance of flipping value per step')
    ax.set_title(f'Average Change of Proposition Value per step ({dir_name})')

    plt.show()


def rollout_proposition_and_reward(dir_name, num_agents=5):
    all_oo_states = []
    all_rewards = []
    for a in range(num_agents):
        file = open(f'./images/{dir_name}/agent-{a}-info.pkl', 'rb')
        info = pickle.load(file)
        all_oo_states.append(info['oo_states'])
        all_rewards.append(info['rewards'])

    print(all_rewards)


if __name__ == "__main__":
    # proposition_distribution()
    # proposition_cooccurence()
    # state_distribution()
    # rollout_proposition_changes(dir_name="expert_rollouts", num_agents=534)
    # rollout_proposition_changes(dir_name="random_rollouts")
    rollout_proposition_and_reward(dir_name="expert_rollouts", num_agents=534)
    # rollout_proposition_and_reward(dir_name="random_rollouts")
