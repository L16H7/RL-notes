import numpy as np
import random

random.seed(42)

# Define the grid world
GRID_LENGTH = 5
TOTAL_STATES = GRID_LENGTH * GRID_LENGTH
ACTION_SPACE = 4  # Up, Right, Down, Left

# Define rewards
STEP_REWARD = -1
GOAL_A_REWARD = 1
GOAL_A_POSITION = (2, 2)
GOAL_B_REWARD = 10
GOAL_B_POSITION = (2, 4)

CLIFF_REWARD = -10
CLIFF_POSITIONS = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

OBSTACLE_POSITIONS = [(1, 1), (2, 1), (2, 3)]

# Define transition probabilities
NOISE = 0.0
PROBABILITY_SUCCESSFUL_ACTION = 1 - NOISE
PROBABILITY_FAILED_ACTION = NOISE
GAMMA = 0.99

# Define actions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

# Create the grid world
GRID_WORLD_REWARDS = np.zeros((GRID_LENGTH, GRID_LENGTH))
GRID_WORLD_REWARDS.fill(STEP_REWARD)
GRID_WORLD_REWARDS[GOAL_A_POSITION[0], GOAL_A_POSITION[1]] = GOAL_A_REWARD
GRID_WORLD_REWARDS[GOAL_B_POSITION[0], GOAL_B_POSITION[1]] = GOAL_B_REWARD
for obstacle_position in OBSTACLE_POSITIONS:
    GRID_WORLD_REWARDS[obstacle_position[0], obstacle_position[1]] = 0
for cliff_position in CLIFF_POSITIONS:
    GRID_WORLD_REWARDS[cliff_position[0], cliff_position[1]] = CLIFF_REWARD

def state_to_coord(state):
    return state // GRID_LENGTH, state % GRID_LENGTH

def coord_to_state(x, y):
    return x * GRID_LENGTH + y

def get_next_state(state, action):
    initial_x, initial_y = state_to_coord(state)

    x, y = state_to_coord(state)
    if action == UP:
        x = max(0, x - 1)
    elif action == RIGHT:
        y = min(GRID_LENGTH - 1, y + 1)
    elif action == DOWN:
        x = min(GRID_LENGTH - 1, x + 1)
    elif action == LEFT:
        y = max(0, y - 1)

    if (x, y) in OBSTACLE_POSITIONS:
        return coord_to_state(initial_x, initial_y)
        
    return coord_to_state(x, y)

def get_reward(state):
    x, y = state_to_coord(state)
    return GRID_WORLD_REWARDS[x][y]

def epsilon_greedy_policy(Q, state, epsilon):
    rng = random.random()
    if rng < epsilon:
        return random.randint(0, ACTION_SPACE - 1)
    else:
        return np.argmax(Q[state])

def q_learning(num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((TOTAL_STATES, ACTION_SPACE))
    
    for episode in range(num_episodes):
        state = random.randint(0, TOTAL_STATES - 1)
        while get_reward(state) == GOAL_A_REWARD or get_reward(state) == GOAL_B_REWARD:
            state = random.randint(0, TOTAL_STATES - 1)
        
        while True:
            action = epsilon_greedy_policy(Q, state, epsilon)

            next_state = None
            if random.random() < PROBABILITY_FAILED_ACTION:
                failed_action = random.randint(0, ACTION_SPACE - 1)
                next_state = get_next_state(state, failed_action)
            else:
                next_state = get_next_state(state, action)

            reward = get_reward(next_state)
            
            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            if reward == GOAL_A_REWARD or reward == GOAL_B_REWARD:
                break
            
            state = next_state
    
    return Q

def print_policy(policy):
    actions = ['^', '>', 'v', '<']
    for i in range(GRID_LENGTH):
        for j in range(GRID_LENGTH):
            state = coord_to_state(i, j)
            if get_reward(state) == GOAL_A_REWARD:
                print('1', end=' ')
            elif get_reward(state) == GOAL_B_REWARD:
                print('10', end=' ')
            elif GRID_WORLD_REWARDS[i, j] == 0:
                print('X', end=' ')
            elif GRID_WORLD_REWARDS[i, j] == CLIFF_REWARD:
                print('C', end=' ')
            else:
                print(actions[np.argmax(policy[state])], end=' ')
        print()

# Run Q-learning
num_episodes = 10000
Q = q_learning(num_episodes, alpha=0.5, gamma=GAMMA)

print("Learned Policy:")
# print(Q)
print_policy(Q)

# print("\nQ-values:")
# print(Q.reshape(GRID_SIZE, GRID_SIZE, NUM_ACTIONS))