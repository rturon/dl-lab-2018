# from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle
import time

import numpy as np
import tensorflow as tf
import tensorflow.layers as layers


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)

class CNN(object):

    def __init__(self, num_filters, kernel_size, device='cpu'):
        # inputs
        if device == 'gpu':
            self.device = '/device:GPU:0'
        else:
            self.device = '/cpu:0'
        print('Device:', self.device)

        # with tf.device(self.device):
        self.inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        # first convolutional layer, default values: strides=1, use_bias=True
        conv1 = layers.Conv2D(filters=num_filters,
                              kernel_size=kernel_size,
                              padding="same",
                              activation="relu")
        pooling = layers.MaxPooling2D(pool_size=2, strides=2)
        conv2 = layers.Conv2D(filters=num_filters,
                              kernel_size=kernel_size,
                              padding="same",
                              activation="relu")
        # flatten layer before the dense layers
        flatten = layers.Flatten()
        # dense layers -> first one with relu?
        linear1 = layers.Dense(units=128, activation="relu")
        # second dense layer only computes logits
        linear2 = layers.Dense(units=10, activation=None)

        # define the graph
        self.logits = conv1(self.inputs)
        for layer in [pooling, conv2, pooling, flatten, linear1, linear2]:
            self.logits = layer(self.logits)

        self.out_soft = tf.nn.softmax(self.logits)

        # intialize the variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        #                        log_device_placement=True))
        self.sess.run(init)

    def predict(self, x):
        ''' computes the output of the network for the input X '''
        # with tf.device(self.device):
        y_pred = self.sess.run(self.out_soft(x))

        return y_pred

    def train(self, x_train, y_train, x_valid, y_valid, num_epochs, lr, batch_size):
        ''' trains the network '''

        t0 = time.time()
        # with tf.device(self.device):
        # define the loss
        y_true = tf.placeholder(tf.float32, shape=[None, 10])
        loss = tf.losses.softmax_cross_entropy(y_true, self.logits)

        # define the optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train = optimizer.minimize(loss)

        # define training loop
        num_batches = x_train.shape[0] // batch_size
        val_error = []

        for i in range(num_epochs):
            train_loss = 0
            for b in range(num_batches):
                x_batch = x_train[b*batch_size:b*batch_size+batch_size]
                y_batch = y_train[b*batch_size:b*batch_size+batch_size]

                _, loss_value = self.sess.run([train, loss],
                                              {self.inputs: x_batch,
                                               y_true: y_batch})
                train_loss += loss_value

            # get the loss for the whole epoch
            train_loss /= num_batches

            # last minibatch
            if num_batches*batch_size < x_train.shape[0]:
                x_batch = x_train[num_batches*batch_size:]
                y_batch = y_train[num_batches*batch_size:]

                _, loss_value = self.sess.run([train, loss],
                                              {self.inputs: x_batch,
                                               y_true: y_batch})
                # adapt loss for whole epoch
                train_loss = train_loss * \
                    ((num_batches*batch_size) / x_train.shape[0]) \
                    + loss_value * \
                    ((x_train.shape[0] - num_batches*batch_size)
                     / x_train.shape[0])

            # get validation loss
            # train_loss = self.sess.run(loss, {self.inputs: x_train})
            val_loss = self.sess.run(loss, {self.inputs: x_valid,
                                            y_true: y_valid})
            val_acc = self.accuracy(x_valid, y_valid)
            train_acc = self.accuracy(x_train, y_train)

            print('Epoch: %i Train loss: %.4f Train accuracy %.3f'
                  % (i, train_loss, train_acc))
            print('Epoch: %i Validation loss: %.4f Validation accuracy %.3f'
                  % (i, val_loss, val_acc))
            val_error.append(1-val_acc)

        t1 = time.time()
        print('Time needed for training: %.2fmin' % ((t1-t0)/60))

        return val_error

    def accuracy(self, x, y_true):
        # with tf.device(self.device):
        y_pred = self.sess.run(self.logits, {self.inputs: x})
        y_pred_int = np.argmax(y_pred, axis=1)
        y_true_int = np.argmax(y_true, axis=1)

        # print(y_pred_int)
        # print()
        # print(y_true_int)
        # print()
        correct = np.sum(y_pred_int == y_true_int)
        # print(correct)
        acc = correct / x.shape[0]

        return acc


def train_and_validate(x_train, y_train, x_valid, y_valid,
                       num_epochs, lr, num_filters, kernel_size, batch_size,
                       device='cpu'):
    # TODO: train and validate your convolutional neural networks with the
    # provided data and hyperparameters
    # build network
    model = CNN(num_filters, kernel_size, device)

    # train network
    learning_curve = model.train(x_train, y_train, x_valid, y_valid,
                                 num_epochs, lr, batch_size)

    return learning_curve, model  # TODO: Return the validation error after each epoch (i.e learning curve) and your model


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    test_acc = model.accuracy(x_test, y_test)

    return 1-test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=32, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Filter width and height")
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)

    test_error = test(x_test, y_test, model)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
