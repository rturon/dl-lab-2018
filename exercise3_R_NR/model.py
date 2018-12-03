import tensorflow as tf

class Model:

    def __init__(self, lr, kernel_size, num_kernels, history_length=1):

        # TODO: Define network
        self.inputs = tf.placeholder(tf.float32,
                                shape=[None, 96, 96, history_length])
        conv1 = tf.layers.conv2d(inputs=self.inputs,
                                 filters=num_kernels,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation='relu')

        pooling1 = tf.layers.max_pooling2d(inputs=conv1,
                                           pool_size=2,
                                           strides=2)

        conv2 = tf.layers.conv2d(inputs=pooling1,
                                 filters=num_kernels,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation='relu')

        pooling2 = tf.layers.max_pooling2d(inputs=conv2,
                                           pool_size=2,
                                           strides=2)

        flat = tf.layers.flatten(pooling2)
        linear1 = tf.layers.dense(inputs=flat,
                                  units=128,
                                  activation='relu')
        self.y_pred = tf.layers.dense(inputs=linear1,
                                 units=5,
                                 activation=None)

        # TODO: Loss and optimizer
        # first get a placeholder for the targets
        self.targets = tf.placeholder(tf.float32, shape=[None, 5])
        self.loss = tf.losses.mean_squared_error(self.targets, self.y_pred)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.train = optimizer.minimize(self.loss)

        # TODO: Start tensorflow session
        # why??
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.saver = tf.train.Saver()

    def accuracy(self, X, y_true):
        self.y_pred = self.sess.run(self.y_pred)

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
