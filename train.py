from gym import make
import random
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

from collections import deque

import warnings
warnings.filterwarnings("ignore")

GAMMA = 0.95
device = torch.device("cpu")


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1 ) % self.capacity

    def sample(self, batch_size):
        return list(zip(*random.sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


def transform_state(state):
    return np.array(state)


class A2C:
    def __init__(self, state_dim, action_dim, capacity, batch_size, low, high, upd_rate):
        self.gamma = GAMMA
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = Memory(capacity)
        self.batch_size = batch_size
        self.low = low
        self.high = high
        self.upd_rate = upd_rate
        self.actor = None
        self.critic = None
        self.build()

    def build(self):
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 2))

        self.actor.apply(init_weights)
        self.actor.to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

        self.critic.apply(init_weights)
        self.critic.to(device)

        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.0003)

    def update_critic(self, estimate, target):
        loss = F.mse_loss(estimate, target)
        self.optimizer_critic.zero_grad()
        loss.backward()

        for param in self.critic.parameters():
            param.grad.data.clamp_(-20, 20)
        self.optimizer_critic.step()

    def update_actor(self, l_p, errors):
        loss = -torch.mm(l_p.T, errors) / l_p.size(0)
        self.optimizer_actor.zero_grad()
        loss.backward()

        for param in self.actor.parameters():
            param.grad.data.clamp_(-20, 20)
        self.optimizer_actor.step()

    def update(self, batch):
        state, action, reward, next_state, done, l_p = batch
        state = torch.cat(state)
        reward = torch.cat(reward)
        next_state = torch.cat(next_state)
        done = torch.cat(done)
        l_p = torch.cat(l_p).unsqueeze(1)

        values = self.critic(state)
        next_value = torch.zeros(next_state.size(0), device=device).unsqueeze(1)
        with torch.no_grad():
            next_value[~done] = self.critic(next_state)[~done]
        target = reward.unsqueeze(1) + self.gamma * next_value

        error = target - values
        self.update_critic(values, target)
        self.update_actor(l_p, error.detach())

    def act(self, state):
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        params = self.actor(state)
        m = 2*F.tanh(params[0][0])
        s = F.softplus(params[0][1])
        norm = Normal(m, s)
        action = norm.sample()
        action = action.unsqueeze(0)
        l_p = norm.log_prob(action)
        action.clamp_(self.low, self.high)
        return action, l_p

    def save(self):
        torch.save(self.actor, "agent.pkl")

    def test(self):
        local_env = make("Pendulum-v0")
        np.random.seed(9)
        local_env.seed(9)
        episodes = 100
        reward = []
        for i in range(episodes):
            state = local_env.reset()
            total_reward = 0
            done = False
            while not done:
                state = torch.tensor(state).to(device).float().unsqueeze(0)
                action, l_p = algo.act(state)
                next_state, r, done, _ = local_env.step(action)
                total_reward += r
                state = next_state
            reward.append(total_reward)
        return reward




if __name__ == "__main__":
    env = make("Pendulum-v0")
    np.random.seed(9)
    env.seed(9)
    low = env.action_space.low[0]
    high = env.action_space.high[0]
    algo = A2C(state_dim=3, action_dim=1, capacity = 40, batch_size = 40, low = low , high = high, upd_rate = 40)
    episodes = 20000
    total_steps = 0
    score = deque(maxlen = 60)
    rew = -600
    t_score = -2000
    for i in range(episodes):
        state = env.reset()
        state = torch.tensor(state).to(device).float().unsqueeze(0)
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action, l_p = algo.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            total_steps += 1
            modified_reward = reward/8 + 1

            next_state = torch.tensor(next_state).to(device).float().unsqueeze(0)
            modified_reward = torch.tensor(modified_reward).to(device).float().unsqueeze(0)
            done = torch.tensor(done).to(device).unsqueeze(0)

            algo.buffer.push((state, action, modified_reward, next_state, done, l_p))

            if total_steps >= algo.batch_size and total_steps % algo.upd_rate == 0:
                algo.update(algo.buffer.sample(algo.batch_size))

            state = next_state

        score.append(total_reward)

        mean = np.mean(score)
        if total_reward > -250  and mean > -370:
            #rew = mean
            test = algo.test()
            br = np.mean(test)
            if br > t_score:
                algo.save()
                t_score = br
            print(f"episode:{i + 1}, reward: {total_reward}, mean: {mean}, test: {br}")
            if br > -150:
                algo.save()
                break

        else:
            print(f"episode:{i + 1}, reward: {total_reward}, mean: {mean}")
            
    t = algo.test()
    print(f"mean: , {np.mean(t)}, test_reward:, {t}")