import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    # classes = np.unique(labels)
    classes = np.array(range(5))
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')


def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    else:
        return STRAIGHT                                      # STRAIGHT = 0

def actions_to_ids(a):
    action_ids = np.zeros(a.shape[0], dtype=int)

    for i in range(a.shape[0]):
        action_ids[i] = action_to_id(a[i])

    return action_ids

def action_distribution(y):

    action_counter = np.zeros(5)

    # action_ids = actions_to_ids(y)
    # unfortunately the next row does not work if not all actions are in the
    # sample
    # _, action_counter = np.unique(action_ids, return_counts=True)
    for i in range(y.shape[0]):
        action_counter[y[i] == 1] += 1

    return action_counter
