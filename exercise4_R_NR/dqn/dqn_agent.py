import tensorflow as tf
import numpy as np
from dqn.replay_buffer import ReplayBuffer

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, discount_factor=0.99, batch_size=64, epsilon=0.05):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         ########################################################################
         TD here for using as new target R + discount_factor * Q(S', A')
         off-policy -> use old data collected on other policy, too
         #######################################################################
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            discount_factor: gamma, discount factor of future rewards.
            batch_size: Number of samples per batch.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
        """
        self.Q = Q
        self.Q_target = Q_target

        self.epsilon = epsilon

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        # define replay buffer
        self.replay_buffer = ReplayBuffer(use_manual_data=False)

        # Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def train(self, state, action, next_state, reward, terminal, collect_data_first=False):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """



        # add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        # if the ReplayBuffer should be filled up first, then the train step is done here
        if collect_data_first and len(self.replay_buffer._data.states) < self.batch_size:
            print("No training yet. Filling up replay buffer..")

            # return 0 for loss and q_values
            return 0, [0,0]

        # If the ReplayBuffer should not be filled up or is full enough, do the following
        else:
            # get a random batch from the ReplayBuffer
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = \
                self.replay_buffer.next_batch(self.batch_size)

            batch_targets = np.zeros((self.batch_size))

            for i in range(self.batch_size):
                # if a state is a final state, only use the direct reward
                if batch_dones[i]:
                    batch_targets[i] = batch_rewards[i]
                # otherwise comput the td_target
                else:
                    td_target = batch_rewards[i] + self.discount_factor * \
                        np.max(self.Q_target.predict(self.sess, [batch_next_states[i]]))
                    batch_targets[i] = td_target

            # update Q network
            loss = self.Q.update(self.sess, batch_states, batch_actions, batch_targets)
            # get predictions to check q-values -> e.g. are they diverging?
            q_preds = self.Q.predict(self.sess, batch_states)

            # update target network
            self.Q_target.update(self.sess)

        return loss, q_preds


    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # take greedy action (argmax)
            action_id = np.argmax(self.Q.predict(self.sess, [state]))
            # print("Deterministic action:", action_id)
        else:

            # sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # for carracing:
            if self.num_actions == 5:
                action_id = np.random.choice(range(5), p=[0.32, 0.09, 0.09, 0.4, 0.1])
            # for cartpole
            action_id = np.random.randint(self.num_actions)
            # print("Explorative action:", action_id)

        return action_id


    def load(self, file_name):
        self.saver.restore(self.sess, file_name)
