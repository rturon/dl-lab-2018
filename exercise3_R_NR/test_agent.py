from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *
from train_agent import preprocess


def run_episode(env, agent, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0

    state = env.reset()
    while True:

        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        state = preprocess(state.reshape((1, state.shape[0], state.shape[1], state.shape[2])))

        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        y_pred = agent.sess.run(agent.y_pred, {agent.inputs: state})[0]

        def unhot(y):
            if np.argmax(y) == 0: return np.array([0.0, 0.0, 0.0])
            elif np.argmax(y) == 1: return np.array([-1.0, 0.0, 0.0])
            elif np.argmax(y) == 2: return np.array([1.0, 0.0, 0.0])
            elif np.argmax(y) == 3: return np.array([0.0, 1.0, 0.0])
            elif np.argmax(y) == 4: return np.array([0.0, 0.0, 0.2])

        a = unhot(y_pred)

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    lr = 0
    ks = 3
    num_kernels = 16
    agent = Model(lr, ks, num_kernels, history_length=1)
    agent.load("models/agent.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
