from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid

def preprocess(X, history_length=1):

    X_gray = rgb2gray(X)

    X_pre = np.zeros((X.shape[0], X.shape[1], X.shape[2], history_length))
    for i in range(X.shape[0] - history_length):
        X_pre[i, :, :, :] = X_gray[i:i + history_length].transpose(1, 2, 0)

    return X_pre

def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot()
    #    useful and you may want to return X_train_unhot ... as well.
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    # for history_length 1 this was it, otherwise we need to perfomr some extra
    # steps
    # preprocess input data
    X_train_gray = preprocess(X_train, history_length)
    X_valid_gray = preprocess(X_valid, history_length)
    # discretize actions
    y_discrete_train = actions_to_ids(y_train)
    y_discrete_valid = actions_to_ids(y_valid)
    # create one-hot vectors
    y_one_hot_train = one_hot(y_discrete_train)
    y_one_hot_valid = one_hot(y_discrete_valid)


    return X_train_gray, y_one_hot_train, X_valid_gray, y_one_hot_valid


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard",
                history_length=1):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")


    # TODO: specify your neural network in model.py
    ks = 5
    num_kernels = 32
    agent = Model(lr, ks, num_kernels, history_length=history_length)

    tensorboard_eval = Evaluation(tensorboard_dir)

    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    def sample_minibatch(X, y, batch_size):
        batch_start = np.random.randint(X_train.shape[0] - batch_size)
        X_minibatch = X[batch_start: batch_start + batch_size]
        y_minibatch = y[batch_start: batch_start + batch_size]

        return X_minibatch, y_minibatch

    def create_probability_mask(y):
        distr = "uniform"
        num_classes = 5
        # get discrete actions
        # y_discrete = actions_to_ids(y)
        action_counts = action_distribution(y)
        # get the probabilities from the frequency of the classes
        if distr == "uniform":
            for i in range(action_counts.shape[0]):
                if action_counts[i] == 0:
                    num_classes -= 1
                else:
                    action_counts[i] = 1/action_counts[i]
            action_counts = action_counts * 1/num_classes

        # get a mask with the probabilities for choosing each data point
        mask = np.zeros(y.shape[0])
        for i in range(5):
            mask[y[:,i] == 1] = action_counts[i]

        return mask


    def sample_minibatch_with_uniform_actions(X, y, batch_size, mask):

        # now sample indices according to the probabilities
        indices = np.arange(y.shape[0])

        batch_inds = np.random.choice(indices, size=batch_size, p=mask)
        # create batch
        X_batch = X[batch_inds,...]
        y_batch = y[batch_inds,...]
        # print(action_distribution(y_batch))

        return X_batch, y_batch

        # since 0 (straight) is 10 times more likely than any other action
        # I choose any
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    #
    # create probability mask for the sampling
    prob_mask = create_probability_mask(y_train)
    # training loop
    for i in range(n_minibatches):
        X_batch, y_batch = sample_minibatch_with_uniform_actions(X_train,
                                                                 y_train,
                                                                 batch_size,
                                                                 prob_mask)
        _, train_loss = agent.sess.run([agent.train, agent.loss],
                                       {agent.inputs: X_batch,
                                        agent.targets: y_batch})
        val_loss = agent.sess.run(agent.loss, {agent.inputs: X_valid,
                                               agent.targets: y_valid})
        if i % 10 == 0:
            train_acc = agent.accuracy(X_batch, y_batch)
            val_acc = agent.accuracy(X_valid, y_valid)
            print('Step %i, Training Loss: %.4f, Validation Loss: %.4f' %(i, train_loss, val_loss))
            print('Train accuracy: %.2f, Validation accuracy: %.2f' %(train_acc * 100,
                                                                  val_acc * 100))
            tensorboard_eval.write_episode_data(i, {"loss": train_loss,
                                                    "val_loss": val_loss})

    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    HISTORY_LENGTH = 1
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data_10000")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train,
                                                       X_valid, y_valid,
                                                       history_length=HISTORY_LENGTH)

    # train model (you can change the parameters!)
    # train_model(X_train, y_train, X_valid, y_valid, n_minibatches=100000, batch_size=64, lr=0.0001)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=201,
                batch_size=128, lr=0.0001, history_length=HISTORY_LENGTH)
