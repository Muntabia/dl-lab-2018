import tensorflow as tf
import numpy as np
from dqn.replay_buffer import ReplayBuffer

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, discount_factor=0.99, batch_size=64, epsilon=0.95,
                 exploration_type='e-annealing', learning_type='dq'):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.
         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            discount_factor: gamma, discount factor of future rewards.
            batch_size: Number of samples per batch.
            epsilon: Chance to sample a random action. Float between 0 and 1.
        """
        self.Q = Q      
        self.Q_target = Q_target
        
        self.epsilon = epsilon
        self.exploration_type = exploration_type
        self.learning_type = learning_type

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        # 2. sample next batch and perform batch update:
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        #       2.1 compute td targets: 
        #              td_target =  reward + discount * argmax_a Q_target(next_state_batch, a)
        if self.learning_type == 'q':
            """q learning"""
            targets = batch_rewards.astype(np.float32)
            targets[np.logical_not(batch_dones)] += self.discount_factor * np.max(self.Q_target.predict(self.sess, batch_next_states), axis=1)[np.logical_not(batch_dones)]
        else:
            """double q learning"""
            q_actions = np.argmax(self.Q.predict(self.sess, batch_next_states), axis=1)
            targets = batch_rewards
            targets[np.logical_not(batch_dones)] += self.discount_factor * self.Q_target.predict(self.sess, batch_next_states)[np.arange(self.batch_size), q_actions][np.logical_not(batch_dones)]
        #       2.2 update the Q network
        #              self.Q.update(...)
        loss = self.Q.update(self.sess, batch_states, batch_actions, targets)
        #       2.3 call soft update for target network
        #              self.Q_target.update(...)
        self.Q_target.update(self.sess)

        return loss


    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or (self.exploration_type=='random' and r < self.epsilon):
            # TODO: take greedy action (argmax)
            a_pred = self.Q.predict(self.sess, [state])
            action_id = np.argmax(a_pred)
        else:
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            if self.exploration_type=='e-annealing' and r > self.epsilon:
                a_pred = self.Q.predict(self.sess, [state])
                action_id = np.argmax(a_pred)

                if self.epsilon > 0.05:
                    self.epsilon *= 0.995
            elif self.exploration_type=='boltzmann':
                tau = 0.5
                action_id = self.Q.boltzmann(self.sess, [state], tau)
            else:
                action_id = np.random.randint(0, self.num_actions)
        return action_id

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)