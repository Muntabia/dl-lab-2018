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
import utils

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=True, max_timesteps=10000, history_length=0, manual=False):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = utils.EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    env.viewer.window.on_key_press = utils.key_press
    env.viewer.window.on_key_release = utils.key_release
    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    while True:
        #skip intro zoom frames
        if step < 48:
            step += 1
            env.step(utils.id_to_action(0))
            continue
        
        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        if do_training and manual:
            action_id = utils.manual_action
        else:
            action_id = agent.act(state, deterministic)
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
        if do_training and (next_state.sum() > 5250): #track out of sight
            print('Track gone:', next_state.sum())
            break
        
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, max_timesteps, skip_frames=0, history_length=0, model_dir="./models_carracing",
                 tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"),
                             ["episode_reward", "straight", "left", "right", "accel", "brake"])

    for i in range(num_episodes):
        print("episode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps
        #(otherwise the car will spend most of the time out of the track)
        #max_timesteps_reduced = int(np.max([500, max_timesteps * i / num_episodes]))
        max_timesteps_reduced = max_timesteps
        drive_manually = i < 1
        stats = run_episode(env, agent, max_timesteps=max_timesteps_reduced, deterministic=False,
                            skip_frames=skip_frames, do_training=True, manual=drive_manually)

        if not drive_manually:
            # epsilon anneal
            agent.anneal()

        # write data
        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "straight" : stats.get_action_usage(utils.STRAIGHT),
                                                      "left" : stats.get_action_usage(utils.LEFT),
                                                      "right" : stats.get_action_usage(utils.RIGHT),
                                                      "accel" : stats.get_action_usage(utils.ACCELERATE),
                                                      "brake" : stats.get_action_usage(utils.BRAKE)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time
        if i % 10 == 0 and i != 0:
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
    sf = 3
    num_actions = 5

    # TODO: Define Q network, target network and DQN agent
    Q = CNN(hl, num_actions)
    Q_target = CNNTargetNetwork(hl, num_actions)
    agent = DQNAgent(Q, Q_target, num_actions, exploration_type='e-annealing', #'boltzmann'
                     discount_factor=0.95,
                     act_random_probability=[4, 6, 6, 12, 1])
    #agent.load(os.path.join("./models_carracing", "dqn_agent.ckpt")) #continue training
    train_online(env, agent, num_episodes=1000, max_timesteps=10000, skip_frames=sf, history_length=hl, model_dir="./models_carracing")
