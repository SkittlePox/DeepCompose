import os
import pickle
import random

import matplotlib.pyplot as plt
from visgrid.envs import GridworldEnv, TaxiEnv
from visgrid.agents.expert import TaxiExpert


def save_rollouts(marker, images, oo_states, states, rewards, expert=False):
    if not expert:
        directory = "../images/random_rollouts"
    else:
        directory = "../images/expert_rollouts"

    os.mkdir(f"{directory}/agent-{marker}")
    for i in range(len(images)):
        plt.imsave(f"{directory}/agent-{marker}/{i}.png", images[i])

    info_dict = {"oo_states": oo_states, "states": states, "rewards": rewards}

    file = open(f'{directory}/agent-{marker}-info.pkl', 'wb')
    pickle.dump(info_dict, file)


def generate_random_states(env, num_states, save=True):
    all_images = []
    all_oo_states = {}
    for i in range(num_states):
        ob, _ = env.reset()
        ob = env.get_observation()
        oo_state = env.get_oo_state()

        all_images.append(ob)
        all_oo_states[f"state-{i}"] = oo_state

        if save:
            plt.imsave(f"../images/random_states/state-{i}.png", ob)
    if save:
        file = open('../images/random_states.pkl', 'wb')
        pickle.dump(all_oo_states, file)


def generate_rollouts(env, agent=None, max_steps=1000, min_rollouts=100, save=True):
    imsum = 0
    all_images = []
    all_states = []
    all_oo_states = []
    all_rewards = []

    def rollout():
        ob, info = env.reset()
        oo_state = env.get_oo_state()
        images = [ob]
        oo_states = [oo_state]
        states = [info['state']]
        rewards = [0]

        n_steps = 0
        while n_steps < max_steps:
            if agent is not None:
                action = agent.act()
            else:
                action = random.randint(0, 4)
            ob, reward, terminal, _, info = env.step(action)
            images.append(ob)
            oo_states.append(env.get_oo_state())
            states.append(info['state'])
            rewards.append(reward)
            n_steps += 1
            if terminal:
                break

        return images, oo_states, states, rewards

    rollout_num = 0

    while imsum < min_rollouts:
        images, oo_states, states, rewards = rollout()
        imsum += len(images)
        all_images.append(images)
        all_oo_states.append(oo_states)
        all_states.append(states)
        all_rewards.append(rewards)
        if save:
            save_rollouts(rollout_num, images, oo_states, states, rewards, expert=agent is not None)
            rollout_num += 1

    return all_images, all_oo_states, all_states


env = TaxiEnv(exploring_starts=True,
              terminate_on_goal=True,
              depot_dropoff_only=False,
              should_render=True,
              dimensions=TaxiEnv.dimensions_5x5_to_64x64)

expert = TaxiExpert(env)

generate_rollouts(env, agent=None, min_rollouts=5000, save=True)
# generate_random_states(env, num_states=200)
