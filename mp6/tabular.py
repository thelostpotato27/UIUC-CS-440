import math

import gym
import numpy as np
import torch

import utils
from policies import QPolicy

# Modified by Mohit Goyal (mohit@illinois.edu) on 04/20/2022

class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update 
        @param gamma: discount factor
        @param model (optional): Load a saved table of Q-values for each state-action
            model = np.zeros(self.buckets + (actionsize,))
            
        """
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env
        self.buckets = buckets
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma
        self.model = np.zeros(self.buckets + (self.actionsize,)) if model is None else model
        # print("model size")
        # print(self.buckets, self.actionsize)
        # print(self.model)
        self.Num_dict = dict()

    def discretize(self, obs):
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation  
        """
        upper_bounds = [self.env.observation_space.high[0], 5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        # print("qvals check")
        discrete_state = self.discretize(states[0])
        # print("states")
        # print(states[0])
        # print(discrete_state)
        model_shape = self.model.shape
        vals = np.zeros((1,model_shape[-1]))
        
        
        for i in range(model_shape[-1]):
            vals[0][i] = self.model[discrete_state + (i,)]
        # print("qvals")
        # print(vals)
        return vals


    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        discrete_state = self.discretize(state)
        q_vals = self.model[discrete_state+ (action,)]
        # print("state")
        # print(state)
        # print("discrete states")
        # print(discrete_state)
        # print("q_vals")
        # print(q_vals)


        d_next_state = self.discretize(next_state)
        # print("next_state info")
        # print(next_state)
        if (done == True):
            target = reward
        else:
            # print("td step")
            # print(self.model)
            model_shape = self.model.shape
            target = reward + self.gamma*max(self.model[d_next_state+ (i,)] for i in range(model_shape[-1]))
            # for i in range(model_shape[-1]):
            #     target = reward+ self.gamma*max(self.model[d_next_state+ (1,)])

        self.model[discrete_state + (action,)] = q_vals + self.lr*(target - q_vals)
        loss = (q_vals - target)**2
        return loss


    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    # env.reset(seed=42) # seed the environment
    # np.random.seed(42) # seed numpy
    # import random
    # random.seed(42)

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(3,8,3,8), actionsize=actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'tabular.npy')
