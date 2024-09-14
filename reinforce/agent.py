from typing import Any
from neural_network import PolicyNetwork
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

class Agent:
    def __init__(self, input_dim, n_actions, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(input_dim=input_dim, n_actions=n_actions)
        self.log_probs = []
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0003)

    def choose_action(self, state):
        logits = self.policy(state)
        #softmax
        probs = F.softmax(logits, dim=-1) #4actions
        # sample action
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        self.log_probs.append(log_prob)
        return action.item()

    def learn(self, rewards):
        '''
        del log prob * G (returns)
        
        turn rewards into G (returns)
        
        '''
        'G2 = r2 + gamma * r3 + gamm^2 r4 + .....'
        # returns = []
        # for i in range(len(rewards)):
        #     G = 0
        #     for j in range(i, len(rewards)):
        #         G += rewards[j] * gamma
        #         gamma *= gamma
        #     returns.append(G)
        self.optimizer.zero_grad()

        
        returns = []
        G_t = 0
        for reward in rewards[::-1]:
            G_t = reward + self.gamma * G_t
            returns.insert(0, G_t)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            print(f"{log_prob=}, {G=}")
            loss += -log_prob * G
            
            # bad_action 0.7 bad return -1
            # -(-.7) = 0.7
            # good action 0.7 good return 1
            # -(.7) = -0.7

            # bad_action 0.1 and bad returns -1
            # -(-.1) = 0.1

        self.log_probs = []

        loss.backward()
        self.optimizer.step()
            

