import pickle
import numpy as np

from train_agent import read_data
from utils import action_distribution, actions_to_ids, one_hot

if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data_10000")
    actions_ids = actions_to_ids(y_train)
    print(actions_ids)
    actions_one_hot = one_hot(actions_ids)
    action_counts = action_distribution(actions_one_hot)

    print('Distribution of actions:')
    print(action_counts)
