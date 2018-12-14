from __future__ import print_function

import sys
sys.path.append("../")
import os
import gym
from dqn.dqn_agent import DQNAgent
from train_carracing import run_episode
from dqn.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped
    num_actions = 5
    sf = 0
    history_length =  0

    #TODO: Define networks and load agent
    # ....
    Q = CNN(history_length, num_actions)
    Q_target = CNNTargetNetwork(history_length, num_actions)
    agent = DQNAgent(Q, Q_target, num_actions)
    agent.load(os.path.join("./models_carracing", "dqn_agent.ckpt"))

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True, history_length=history_length, skip_frames=sf)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
