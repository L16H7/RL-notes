import random
import torch
from neural_network import DQN
import torch.optim as optim
import numpy as np

class Agent:
    def __init__(self, epsilon, input_dim, n_actions, gamma=0.99) -> None:
        self.epsilon = epsilon
        self.model = DQN(input_dim=input_dim, n_actions=n_actions)
        self.target_model = DQN(input_dim=input_dim, n_actions=n_actions)
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state):
        '''
        choose action based on epsilon greedy
        INPUT:
            state: state of agent
        OUTPUT:
            action: action of the given state
        '''
        if random.random() > self.epsilon:
            action_values = self.model.forward(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(action_values)
            action = action.item()
            return action

        return random.choice([0, 1, 2, 3])

    def learn(self, memory_buffer, replace):
        print('learn')
        states, actions, rewards, next_states, dones = memory_buffer
        # randomize the indices
        indices = np.random.permutation(128)

        # batch the data into 32 samples
        batch_size = 32
        num_batches = 128 // batch_size

        for i in range(num_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = rewards[batch_indices]
            batch_next_states = next_states[batch_indices]
            batch_dones = dones[batch_indices]

            batch_states = torch.tensor(batch_states, dtype=torch.float32)
            batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32)
            batch_dones = torch.tensor(batch_dones, dtype=torch.float32)

            # target = r + gamma * max(Q(s_t+1, a))
            next_state_q_values = self.target_model.forward(batch_next_states) # Q_s_t+1
            max_q_values, _ = torch.max(next_state_q_values, axis=1)  # max(Q(s_t+1, a))

            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)

            # target = r + gamma * max(Q(s_t+1, a))
            target = batch_rewards + self.gamma * max_q_values * (1 - batch_dones.float())

            # Q(s)
            current_q_values = self.model.forward(batch_states)
            
            # Q(s, a) #.gather(1, actions)?
            q_values = current_q_values[1, batch_actions]

            loss = torch.nn.MSELoss()(q_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if replace:
                self.target_model.load_state_dict(self.model.state_dict())
                