from __future__ import print_function

import gym
from datetime import datetime
import os
import json
from dqn.dqn_agent import DQNAgent
from train_carracing import run_episode
from dqn.networks import *
import numpy as np


np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    # history_length =  0

    #TODO: Define networks and load agent
    # set parameters
    num_kernels = 16
    kernel_size = 5
    history_length=0
    bs = 64
    df = 0.9
    tau = 0.01

    # get state space and number of actions
    state_dim = env.observation_space.shape
    state_dim = (state_dim[0], state_dim[1], history_length+1)
    num_actions = 5

    Q = CNN(state_dim, num_actions, num_kernels, kernel_size)
    # create target network
    Q_target = CNNTargetNetwork(state_dim, num_actions, num_kernels, kernel_size,
                                1e-4, tau)
    # create dqn_agent
    agent = DQNAgent(Q, Q_target, num_actions, df)
    agent.load('./models_carracing/dqn_agent.ckpt')

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False,
                            rendering=True, history_length=history_length)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
