import tensorflow as tf
import numpy as np


class CNN():
    """
    Convolutional Neural Network class for the carracing RL agent
    """
    def __init__(self, state_dim, num_actions, num_kernels, kernel_size, lr=1e-4, stride=1):
        self._build_model(state_dim, num_actions, num_kernels, kernel_size, lr, stride)

    def _build_model(self, state_dim, num_actions, num_kernels, kernel_size, lr, stride):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """

        self.states_ = tf.placeholder(tf.float32, shape=[None, state_dim[0],
                                                         state_dim[1],
                                                         state_dim[2]])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])               # The TD target value

        # two possibilities to create a network:
        # either create a network with two conv layers, same number of kernels and same kernel_size
        if isinstance(num_kernels, int) or isinstance(kernel_size, int) or isinstance(stride, int):
            conv1 = tf.layers.conv2d(self.states_,filters=num_kernels, kernel_size=kernel_size,
                                     strides=1, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1,filters=num_kernels, kernel_size=kernel_size,
                                     strides=1, activation=tf.nn.relu)
            flat = tf.layers.flatten(conv2)

        # or create a network with three conv layers and individual kernel sizes, number of kernels
        # and individual stride size
        else:
            conv1 = tf.layers.conv2d(self.states_, filters=num_kernels[0],
                                     kernel_size=kernel_size[0], strides=stride[0],
                                     activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, filters=num_kernels[1],
                                     kernel_size=kernel_size[1], strides=stride[1],
                                     activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(conv2, filters=num_kernels[2],
                                      kernel_size=kernel_size[2], strides=stride[2],
                                      activation=tf.nn.relu)
            flat = tf.layers.flatten(conv3)

        # one fully connected layer after the conv layers
        fc1 = tf.layers.dense(flat, 100, tf.nn.relu)
        self.predictions = tf.layers.dense(fc1, num_actions)

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        # get index of actions chosen when predictions flattened
        # -> first part: get first index of each row in flattened array
        # -> + actions: add correct column index
        self.gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        prediction = sess.run(self.predictions, {self.states_: states})
        return prediction


    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class CNNTargetNetwork(CNN):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, state_dim, num_actions, num_kernels, kernel_size, lr=1e-4, tau=0.01, stride=1):
        super(CNNTargetNetwork, self).__init__(state_dim, num_actions, num_kernels,
                                            kernel_size, lr, stride)

        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
          sess.run(op)


class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4):
        self._build_model(state_dim, num_actions, hidden, lr)

    def _build_model(self, state_dim, num_actions, hidden, lr):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """

        self.states_ = tf.placeholder(tf.float32, shape=[None, state_dim])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])               # The TD target value

        # network
        fc1 = tf.layers.dense(self.states_, hidden, tf.nn.relu)
        fc2 = tf.layers.dense(fc1, hidden, tf.nn.relu)
        self.predictions = tf.layers.dense(fc2, num_actions)

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        # get index of actions chosen when predictions flattened
        # -> first part: get first index of each row in flattened array
        # -> + actions: add correct column index
        self.gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        prediction = sess.run(self.predictions, {self.states_: states})
        return prediction


    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4, tau=0.01):
        super(TargetNetwork, self).__init__(state_dim, num_actions, hidden, lr)
        # NeuralNetwork.__init__(self, state_dim, num_actions, hidden, lr)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
          sess.run(op)
