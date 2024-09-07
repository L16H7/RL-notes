import numpy as np


class Memory:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    def store_memory(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
    def get_memory(self):
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.next_states), np.array(self.dones)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
