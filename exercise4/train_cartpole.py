import numpy as np
import gym
import itertools as it
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from dqn.networks import NeuralNetwork, TargetNetwork
from utils import EpisodeStats
from start_tensorboard import TensorBoardTool


def run_episode(env, agent, deterministic, do_training=True, rendering=True, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        
        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)
        if do_training:  
            agent.add(state, action_id, next_state, reward, terminal)
            loss = agent.train()
        stats.step(reward, action_id)

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

 
    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "a_0", "a_1"])
    tensorboard_eval = Evaluation(os.path.join(tensorboard_dir, "eval"), ["episode_reward", "a_0", "a_1"])

    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)

        # epsilon anneal
        agent.anneal()

        # write data
        tensorboard.write_episode_data(i, eval_dict={"episode_reward" : stats.episode_reward,
                                                                "a_0" : stats.get_action_usage(0),
                                                                "a_1" : stats.get_action_usage(1)})

        # TODO: evaluate your agent once in a while for some episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % 100 == 0 and i != 0:
            stats = run_episode(env, agent, deterministic=True, do_training=False)
            tensorboard_eval.write_episode_data(i, eval_dict={"episode_reward": stats.episode_reward,
                                                         "a_0": stats.get_action_usage(0),
                                                         "a_1": stats.get_action_usage(1)})
       
        # store model every 100 episodes and in the end.
        if i % 100 == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))
   
    tensorboard.close_session()


if __name__ == "__main__":

    # start tensorboard without commandline usage
    tb_tool = TensorBoardTool()
    tb_tool.run()

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    game = 1  # which game? cartpole == 1 or mountaincar == 2

    if game == 1:
        env = gym.make("CartPole-v0").unwrapped
        state_dim = 4
        num_actions = 2
        episodes = 1500
        model_dir = "./models_cartpole"
    else:
        env = gym.make("MountainCar-v0").unwrapped
        state_dim = 2
        num_actions = 3
        episodes = 1000
        model_dir = "./models_mountaincar"
    
    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    Q = NeuralNetwork(state_dim, num_actions)
    Q_target = TargetNetwork(state_dim, num_actions)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    DQNAgent = DQNAgent(Q, Q_target, num_actions,
                        replay_buffer_size = 1e4,
                        epsilon = 1.0,
                        epsilon_decay = 0.999)
    # 3. train DQN agent with train_online(...)
    train_online(env, DQNAgent, num_episodes=episodes, model_dir=model_dir)
