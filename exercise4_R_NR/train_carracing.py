# export DISPLAY=:0

import sys
sys.path.append("../")

import numpy as np
import gym
import time
from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import EpisodeStats, action_id_to_action, rgb2gray

STRAIGHT = 0
LEFT = 1
RIGHT = 2
ACCELERATE = 3
BRAKE = 4


def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True,
                rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    q_values = []
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)

    while True:

        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(state, deterministic=deterministic)
        # action = your_id_to_action_method(...)
        action = action_id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            loss, q_preds = agent.train(state, action_id, next_state, reward, terminal)
            q_values.append(np.mean(q_preds))

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps :
            break

        step += 1

    if do_training:
        return stats, loss, q_values

    return stats


def train_online(env, agent, num_episodes, history_length=0,
                 model_dir="./models_carracing", tensorboard_dir="./tensorboard"):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    # added some variables for tracking (loss, mean_q)
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"),
                             ["episode_reward", "straight", "left", "right",
                             "accel", "brake", "loss", "mean_q"])

    # create a list to save all the deterministic rewards during training
    rewards_det = []

    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing
        # max_timesteps (otherwise the car will spend most of the time out of the track)

        # decrease epsilon gradually during training
        epsilons = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

        if i < 20:
            max_timesteps = 250
            agent.epsilon = epsilons[0]
        elif i < 40:
            max_timesteps = 250
            agent.epsilon = epsilons[1]
        elif i < 60:
            max_timesteps = 500
            agent.epsilon = epsilons[2]
        elif i < 80:
            max_timesteps = 500
            agent.epsilon = epsilons[3]
        elif i < 100:
            max_timesteps = 500
            agent.epsilon = epsilons[4]
        # last decrease of epsilon after 100 episodes
        else:
            max_timesteps = 1000
            agent.epsilon = epsilons[5]

        stats, loss, q_values = run_episode(env, agent, max_timesteps=max_timesteps,
                                            skip_frames=3, deterministic=False,
                                            do_training=True, rendering=False)
        mean_q = np.mean(q_values)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE),
                                                      "loss" : loss,
                                                      "mean_q": mean_q
                                                      })

        # evaluate agent with deterministic policy every 10th episode
        if i % 10 == 0:
            stats_val = run_episode(env, agent, max_timesteps=max_timesteps,
                                                            skip_frames=0, deterministic=True,
                                                            do_training=False, rendering=True)
            print("Episode reward deterministic:", stats_val.episode_reward)
            rewards_det.append(stats_val.episode_reward)

        # save agent every 20th episode
        if i % 20 == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

    tensorboard.close_session()
    # print all deterministic rewards again at the end of training
    print("Deterministic rewards:", rewards_det)

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

def create_and_train_agent(num_kernels, kernel_size, lr, history_length,
                           batch_size, num_episodes, epsilon, discount_factor,
                           tau, stride=1, model_dir="./models_carracing"):

    env = gym.make('CarRacing-v0').unwrapped

    state_dim = env.observation_space.shape
    state_dim = (state_dim[0], state_dim[1], history_length+1)
    # only use straight, left, right, accelerate and brake
    num_actions = 5

    # create Q network
    Q = CNN(state_dim, num_actions, num_kernels, kernel_size, lr, stride)
    # create target network
    Q_target = CNNTargetNetwork(state_dim, num_actions, num_kernels, kernel_size,
                                lr, tau, stride)
    print("Creating agent now ..")
    # create dqn_agent
    dqn_agent = DQNAgent(Q, Q_target, num_actions, discount_factor, batch_size, epsilon)
    # reload already trained agent for further training
    # dqn_agent.load('./models_carracing/dqn_agent.ckpt')

    start_time = time.time()
    train_online(env, dqn_agent, num_episodes, history_length, model_dir)
    end_time = time.time()
    print("Time needed for training:", (end_time - start_time)/60, "min")

if __name__ == "__main__":

    # define most parameters here to make changes for testing purposes easier
    # then make one function call that creates the networks and agents and
    # performs the training
    # set parameters
    num_kernels = [32, 64, 64]
    kernel_size = [8, 4, 3]
    lr = 0.001
    history_length=0
    bs = 128
    num_episodes = 500
    epsilon = 0.5
    df = 0.99
    tau = 0.01
    stride = [4, 2, 1]

    # call the function to create and train networks and agent
    create_and_train_agent(num_kernels, kernel_size, lr, history_length, bs,
                           num_episodes, epsilon, df, tau, stride)
