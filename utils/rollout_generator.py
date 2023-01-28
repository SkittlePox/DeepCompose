import os
import pickle
import random

import matplotlib.pyplot as plt
from visgrid.envs import GridworldEnv, TaxiEnv
from visgrid.agents.expert import TaxiExpert


def save_rollouts(images, marker, oo_states=None, states=None):
    os.mkdir(f"../images/rollouts/agent-{marker}")
    for i in range(len(images)):
        plt.imsave(f"../images/rollouts/agent-{marker}/{i}.png", images[i])
    file = open(f'../images/rollouts/agent-{marker}-propositions.pkl', 'wb')
    pickle.dump(oo_states, file)
    file2 = open(f'../images/rollouts/agent-{marker}-states.pkl', 'wb')
    pickle.dump(states, file2)


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


def generate_rollouts(env, agent=None, max_steps=100, min_rollouts=100, save=True):
    imsum = 0
    all_images = []
    all_states = []
    all_oo_states = []

    def rollout():
        ob, info = env.reset()
        oo_state = env.get_oo_state()
        images = [ob]
        oo_states = [oo_state]
        states = [info['state']]

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
            n_steps += 1
            if terminal:
                break

        return images, oo_states, states

    rollout_num = 0

    while imsum < min_rollouts:
        images, oo_states, states = rollout()
        imsum += len(images)
        all_images.append(images)
        all_oo_states.append(oo_states)
        all_states.append(states)
        if save:
            save_rollouts(images, rollout_num, oo_states, states)
            rollout_num += 1

    return all_images, all_oo_states, all_states


env = TaxiEnv(exploring_starts=True,
              terminate_on_goal=True,
              depot_dropoff_only=False,
              should_render=True,
              dimensions=TaxiEnv.dimensions_5x5_to_64x64)

# expert = TaxiExpert(env)

generate_rollouts(env, min_rollouts=1000, save=True)
# generate_random_states(env, num_states=200)
