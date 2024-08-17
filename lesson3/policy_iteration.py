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
NOISE = 0.1
PROBABILITY_SUCCESSFUL_ACTION = 1 - NOISE
PROBABILITY_FAILED_ACTION = NOISE
GAMMA = 0.1

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

def policy_evaluation(policy, V, theta=0.01):
    while True:
        delta = 0
        for state in range(TOTAL_STATES):
            v = V[state]

            action = policy[state]
            next_state = None
            if random.random() < PROBABILITY_FAILED_ACTION:
                failed_action = random.choice([UP, DOWN, LEFT, RIGHT])
                next_state = get_next_state(state, failed_action)
            else:
                next_state = get_next_state(state, action)
            
            reward = get_reward(next_state)

            V[state] = reward + (GAMMA * V[next_state])
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def policy_improvement(policy, V):
    policy_stable = True
    
    for state in range(TOTAL_STATES):
        old_action = policy[state]
        
        best_action = None
        best_value = float('-inf')
        for action in range(ACTION_SPACE):
            next_state = get_next_state(state, action)
            value = get_reward(next_state) + GAMMA * V[next_state]

            if value > best_value:
                best_value = value
                best_action = action

        policy[state] = best_action

        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable

def policy_iteration():
    policy = np.random.choice(ACTION_SPACE, TOTAL_STATES)
    V = np.zeros(TOTAL_STATES)
    
    while True:
        V = policy_evaluation(policy, V)
        policy, policy_stable = policy_improvement(policy, V)
        if policy_stable:
            break
    
    return policy, V

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
                print(actions[policy[state]], end=' ')
        print()

# Run policy iteration
optimal_policy, optimal_value = policy_iteration()

print("Optimal Policy:")
print_policy(optimal_policy)
