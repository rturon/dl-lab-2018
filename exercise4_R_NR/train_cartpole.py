import numpy as np
import gym
import itertools as it
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from dqn.networks import NeuralNetwork, TargetNetwork
from utils import EpisodeStats


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    q_values = []
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            loss, qs = agent.train(state, action_id, next_state, reward, terminal)
            q_values.append(np.mean(qs))

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    if do_training:
        return stats, loss, q_values

    return stats

def train_online(env, agent, num_episodes, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), [
        "episode_reward", "a_0", "a_1", "loss", "mean_q"])


    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True, rendering=False)

        stats, loss, q_values = stats[0], stats[1], stats[2]
        # compute mean of q_values
        mean_q = np.mean(q_values)

        # added some variables for tracking (loss, mean_q)
        tensorboard.write_episode_data(i, eval_dict={
            "episode_reward" : stats.episode_reward,
            "a_0" : stats.get_action_usage(0),
            "a_1" : stats.get_action_usage(1),
            "loss": loss,
            "mean_q": mean_q})

        # terminate training if loss is too high
        if loss > 1000:
            print('Loss diverging: ', loss)
            break

        # test the deterministic policy every 50th episode
        if i % 50 == 0:
            stats_det = run_episode(env, agent, deterministic=True, do_training=False)
            print('Episode reward deterministic:', stats_det.episode_reward)
            print('Action 0 selected:', stats_det.get_action_usage(0))

        # store model every 100 episodes and in the end.
        if i % 100 == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

    tensorboard.close_session()

def create_and_train_agent(lr, epsilon, discount_factor, bs, tau, num_episodes,
                           hidden=20):

    # create environment
    env = gym.make("CartPole-v0").unwrapped

    # get state space and number of actions
    state_dim = 4 # env.observation_space.shape[0]
    num_actions = 2 # env.action_space.n

    # create neural networks
    Q = NeuralNetwork(state_dim=state_dim, num_actions=num_actions,
                     hidden=hidden, lr=lr)
    Q_target = TargetNetwork(state_dim=state_dim, num_actions=num_actions,
                             hidden=hidden, lr=lr, tau=tau)
    # create agent
    agent = DQNAgent(Q, Q_target, num_actions, discount_factor=discount_factor,
                     batch_size=bs, epsilon=epsilon)
    # train agent
    train_online(env, agent, num_episodes=num_episodes)

    # get some final values to compare different networks
    rewards = []
    for i in range(10):
        stats_det = run_episode(env, agent, deterministic=True, do_training=False)
        rewards.append(stats_det.episode_reward)

    return np.mean(rewards)

if __name__ == "__main__":

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater
    # than or equal to 195.0 over 100 consecutive trials.

    # set some parameters that could be interesting for training here and then
    # call a function that creates the networks and the agent and trains them

    # set parameters
    NUM_EPISODES = 170
    EPSILON = 0.05            # default 0.05
    LEARNING_RATE = 0.003   # default 1e-4
    BATCHSIZE = 64             # default 64
    DISCOUNT_FACTOR = 0.9     # default 0.99
    TAU = 0.01                 # default 0.01
    NUM_UNITS = 16           # default 20

    # create networks and agent and train it
    final_rewards = create_and_train_agent(LEARNING_RATE, EPSILON, DISCOUNT_FACTOR, BATCHSIZE,
                           TAU, NUM_EPISODES, hidden=NUM_UNITS)

    print("Mean reward after training:", final_rewards)
