from gym import make
import random
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
#from .train import transform_state

device = torch.device( "cpu")


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/A2C_agent.pkl")
        #self.model.to(device)

    def act(self, state):
        state = torch.tensor(state).to(device).float().unsqueeze(0)
        params = self.model(state)
        m = torch.tanh(params[0][0])
        s = F.softplus(params[0][1])
        norm = Normal(m, s)
        action = norm.sample()
        action = action.unsqueeze(0)
        action.clamp_(-2, 2)
        #return np.array([action.item()])
        return action

    def reset(self):
        pass

if __name__ == "__main__":
    local_env = make("Pendulum-v0")
    np.random.seed(9)
    local_env.seed(9)
    episodes = 100
    reward = []
    agent = Agent()
    for i in range(episodes):
        state = local_env.reset()
        total_reward = 0
        done = False
        while not done:
            #state = torch.tensor(state).to(device).float().unsqueeze(0)
            action = agent.act(state)
            next_state, r, done, _ = local_env.step(action)
            total_reward += r
            state = next_state
        reward.append(total_reward)
    print(np.mean(reward), np.max(reward), "reward", reward)