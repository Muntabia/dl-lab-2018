from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=3):
    episode_reward = 0
    step = 0

    state = env.reset()
    while True:

        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        state = state[np.newaxis, ...]
        state = rgb2gray(state)

        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        # just to make sure that the agent starts driving
        count_accelerate = 0
        max_accelerate = 15
        if step < 5:
            a = [0., 1., 0.]
            count_accelerate += 1
        else:
            a = agent.sess.run(agent.logits_unhot, feed_dict={agent.X: state})
            a = id_to_action(a)
            if a == [0., 1., 0.]:
                count_accelerate += 1
                if count_accelerate > max_accelerate:
                    a = [0., 0., 0.]
        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test

    # TODO: load agent
    lr = 0.0020844121859572885
    num_layers = 2
    num_filters = (45, 23)
    filter_size = (3, 3)
    stride = (2, 1)
    padding = ('same', 'same')
    maxpool = (True, True)
    batch_size = 45
    hl = 1
    agent = Model(hl, num_layers, num_filters, filter_size, stride, padding, maxpool, lr)
    agent.load("models/run0/agent.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)
    # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()

    fname = "./results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(fname, 'w') as fh:
        json.dump(results, fh)

    env.close()
    print('... finished')
