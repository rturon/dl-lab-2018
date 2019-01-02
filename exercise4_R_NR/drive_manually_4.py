from __future__ import print_function

import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
from utils import rgb2gray


def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.2

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0.0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0.0
    if k == key.UP:    a[1] = 0.0
    if k == key.DOWN:  a[2] = 0.0


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()

    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')

def action_to_id(a):
    if all(a == [-1.0, 0.0, 0.0]): return 1              # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return 2             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return 3        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]): return 4             # BRAKE: 4
    else:
        return 0                                    # STRAIGHT = 0

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--collect_data", action="store_true", default=False, help="Collect the data in a pickle file.")

    args = parser.parse_args()

    samples = {
        "states": [],
        "next_states": [],
        "rewards": [],
        "actions": [],
        "dones" : [],
    }

    env = gym.make('CarRacing-v0').unwrapped

    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release


    a = np.array([0.0, 0.0, 0.0]).astype('float32')

    episode_rewards = []
    steps = 0
    while True:
        episode_reward = 0
        state = env.reset()
        while True:

            next_state, r, done, info = env.step(a)
            episode_reward += r

            state_lst = state_preprocessing(state)
            state_lst = np.array(state_lst).reshape(96, 96, 1)

            next_state_lst = state_preprocessing(next_state)
            next_state_lst = np.array(next_state_lst).reshape(96, 96, 1)

            samples["states"].append(state_lst)            # state has shape (96, 96, 3)
            samples["actions"].append(action_to_id(a))     # action now is an integer
            samples["next_states"].append(next_state_lst)
            samples["rewards"].append(r)
            samples["dones"].append(done)

            state = next_state
            steps += 1

            if steps % 800 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("\nstep {}".format(steps))

            if args.collect_data and steps % 1000 == 0:
                print('... saving data')
                store_data(samples, "./manual_data")
                # save_results(episode_rewards, "./results")

            env.render()
            if done:
                break

        episode_rewards.append(episode_reward)

    env.close()
