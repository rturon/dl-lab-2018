import numpy as np


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []


    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)


    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))


def action_id_to_action(action_id):
    '''
    Transforms an action_id between 0 and 4 to an action for the gym CarRacing
    environment. The shape of the returned action is the same as the shape of
    an action of the gym environment.
    '''
    if action_id == 0:
        return np.array([0.0, 0.0, 0.0])    # Straight
    elif action_id == 1:
        return np.array([-1.0, 0.0, 0.0])   # Left
    elif action_id == 2:
        return np.array([1.0, 0.0, 0.0])    # Right
    elif action_id == 3:
        return np.array([0.0, 1.0, 0.0])    # Accelerate
    else:
        return np.array([0.0, 0.0, 0.2])    # Brake

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')
