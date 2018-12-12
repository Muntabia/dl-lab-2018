# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import matplotlib.pyplot as plt
import itertools as it
from utils import EpisodeStats
import utils

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=True, max_timesteps=10000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    
    while True:
        
        if step < 48:
            step += 1
            continue #skip intro zoom frames

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        action_id = agent.act(state, deterministic=False)
        action = utils.id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, max_timesteps, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "straight", "left", "right", "accel", "brake"])

    for i in range(num_episodes):
        print("episode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        max_timesteps_reduced = max_timesteps * i / num_episodes
        stats = run_episode(env, agent, max_timesteps=max_timesteps_reduced, deterministic=False, skip_frames=0, do_training=True)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(utils.STRAIGHT),
                                                      "left" : stats.get_action_usage(utils.LEFT),
                                                      "right" : stats.get_action_usage(utils.RIGHT),
                                                      "accel" : stats.get_action_usage(utils.ACCELERATE),
                                                      "brake" : stats.get_action_usage(utils.BRAKE)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time
        if i % 10 == 0:
            stats = run_episode(env, agent, max_timesteps=1000, deterministic=True, do_training=False)

        if i % 100 == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt")) 

    tensorboard.close_session()

def state_preprocessing(state):
    gray = utils.rgb2gray(state).reshape(96, 96) / 255.0
    return gray

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped
    hl = 0
    num_actions = 5

    # TODO: Define Q network, target network and DQN agent
    Q = CNN(hl, num_actions)
    Q_target = CNNTargetNetwork(hl, num_actions)
    agent = DQNAgent(Q, Q_target, num_actions, exploration_type='e-annealing', #'boltzmann'
                     act_random_probability=[1/9, 2/9, 2/9, 3/9, 1/9])
    
    train_online(env, agent, num_episodes=1000, max_timesteps=10000, history_length=hl, model_dir="./models_carracing")
