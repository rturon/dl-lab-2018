import os
from datetime import datetime
import gym
import json
from dqn.dqn_agent import DQNAgent
from train_cartpole import run_episode
from dqn.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v0").unwrapped

    # load agent
    env = gym.make("CartPole-v0").unwrapped

    # get state space and number of actions
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # create neural networks
    Q = NeuralNetwork(state_dim=state_dim, num_actions=num_actions)
    Q_target = TargetNetwork(state_dim=state_dim, num_actions=num_actions)
    # create agent
    agent = DQNAgent(Q, Q_target, num_actions, discount_factor=0.7)
    agent.load('./models_cartpole/dqn_agent.ckpt')

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
