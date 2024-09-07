'''
Action Space Discrete(4)
0: do nothing

1: fire left orientation engine

2: fire main engine

3: fire right orientation engine

Observation Space
Box([-1.5 -1.5 -5. -5. -3.1415927 -5. -0. -0. ], [1.5 1.5 5. 5. 3.1415927 5. 1. 1. ], (8,), float32)
The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
'''

import numpy as np
import gymnasium as gym
from memory import Memory
from agent import Agent

# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v2", render_mode="human")

memory = Memory()
epsilon = 0.5
agent = Agent(epsilon=epsilon, input_dim=8, n_actions=4)

n_episode = 100
step = 0
for i in range(n_episode):
    state, _ = env.reset()
    done = False

    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        memory.store_memory(state, action, reward, next_state, done)
        total_reward += reward

        state = next_state
        step += 1
        
        if step % 128 == 0:
            memory_buffer = memory.get_memory()

            is_replace_target = step % 1024 == 0
            agent.learn(memory_buffer, is_replace_target)
            memory.clear_memory()

        if i % 10 == 0:
            print(f"Episode {i}/{n_episode}, Total Reward: {total_reward}")
