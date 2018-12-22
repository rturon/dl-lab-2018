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
        self.replay_buffer = ReplayBuffer()

        # Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer

        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        # 2. sample next batch and perform batch update:
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = \
            self.replay_buffer.next_batch(self.batch_size)
        # print('###### Batch states ############')
        # print(batch_states)
        #       2.1 compute td targets:
        #              td_target =  reward + discount * argmax_a Q_target(next_state_batch, a)
        # print(batch_dones)
        target_Qs = np.max(self.Q_target.predict(self.sess, batch_next_states), axis=1)
        batch_targets = batch_rewards + self.discount_factor * target_Qs * batch_dones
        #       2.2 update the Q network
        #              self.Q.update(...)
        loss = self.Q.update(self.sess, batch_states, batch_actions, batch_targets)
        # get predictions to check q-values
        q_preds = self.Q.predict(self.sess, batch_states)
        #       2.3 call soft update for target network
        #              self.Q_target.update(...)
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
            # TODO: take greedy action (argmax)
            #state = state.reshape(-1, )
            action_id = int(np.argmax(self.Q.predict(self.sess, [state]), axis=1))
            # print('Action according to policy: ', action_id)
        else:

            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            ##### for carracing:
            ##### np.random.choice(np.range(self.num_actions), p=[0.1, 0.1, 0.1, 0.1, 0.6])
            action_id = np.random.randint(self.num_actions)
            # print('Exploration: ', action_id)
        # print('Action id:', action_id)
        return action_id


    def load(self, file_name):
        self.saver.restore(self.sess, file_name)
