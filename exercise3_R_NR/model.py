import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, lr, kernel_size, num_kernels, history_length=1):

        # TODO: Define network
        self.inputs = tf.placeholder(tf.float32,
                                shape=[None, 96, 96, history_length])
        conv1 = tf.layers.conv2d(inputs=self.inputs,
                                 filters=num_kernels,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation=tf.nn.relu)

        pooling1 = tf.layers.max_pooling2d(inputs=conv1,
                                           pool_size=2,
                                           strides=2)

        conv2 = tf.layers.conv2d(inputs=pooling1,
                                 filters=num_kernels,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation=tf.nn.relu)

        pooling2 = tf.layers.max_pooling2d(inputs=conv2,
                                           pool_size=2,
                                           strides=2)

        conv3 = tf.layers.conv2d(inputs=pooling2,
                                 filters=num_kernels,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation=tf.nn.relu)

        pooling3 = tf.layers.max_pooling2d(inputs=conv3,
                                           pool_size=2,
                                           strides=2)

        # conv4 = tf.layers.conv2d(inputs=pooling3,
        #                          filters=num_kernels,
        #                          kernel_size=kernel_size,
        #                          padding='same',
        #                          activation=tf.nn.relu)
        #
        # pooling4 = tf.layers.max_pooling2d(inputs=conv4,
        #                                    pool_size=2,
        #                                    strides=2)

        flat = tf.layers.flatten(pooling3)
        linear1 = tf.layers.dense(inputs=flat,
                                  units=100,
                                  activation=tf.nn.relu)
        drop1 = tf.nn.dropout(linear1, 0.8)
        linear2 = tf.layers.dense(inputs=drop1,
                                  units=100,
                                  activation=tf.nn.relu)
        drop2 = tf.nn.dropout(linear2, 0.8)
        self.y_pred = tf.layers.dense(inputs=drop2,
                                 units=5,
                                 activation=None)

        # TODO: Loss and optimizer
        # first get a placeholder for the targets
        self.targets = tf.placeholder(tf.float32, shape=[None, 5])
        self.loss = tf.losses.softmax_cross_entropy(self.targets, self.y_pred)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train = optimizer.minimize(self.loss)

        # TODO: Start tensorflow session
        # why??
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.saver = tf.train.Saver()

    def accuracy(self, X, y_true):
        y_pred = self.sess.run(self.y_pred, {self.inputs: X})

        y_pred_int = np.argmax(y_pred, axis=1)
        y_true_int = np.argmax(y_true, axis=1)

        correct = np.sum(y_pred_int == y_true_int)
        acc = correct / X.shape[0]

        return acc

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
