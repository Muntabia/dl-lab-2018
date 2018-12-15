import numpy as np
from pyglet.window import key

STRAIGHT = 0
LEFT = 1
RIGHT = 2
ACCELERATE = 3
BRAKE = 4

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
        return len(ids[ids == action_id]) / len(ids)

def id_to_action(a):
    if a == LEFT: return [-1.0, 0.0, 0.0]               # LEFT: 1
    elif a == RIGHT: return [1.0, 0.0, 0.0]             # RIGHT: 2
    elif a == ACCELERATE: return [0.0, 1.0, 0.0]        # ACCELERATE: 3
    elif a == BRAKE: return [0.0, 0.0, 0.8]             # BRAKE: 4
    else:
        return [0.0, 0.0, 0.0]                          # STRAIGHT = 0

def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return gray


#manual driving
manual_action = 0
def key_press(k, mod):
    global manual_action
    if k == key.LEFT:  manual_action = 1
    if k == key.RIGHT: manual_action = 2
    if k == key.UP:    manual_action = 3
    if k == key.DOWN:  manual_action = 4

def key_release(k, mod):
    global manual_action
    manual_action = 0
