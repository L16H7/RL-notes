import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 5
CELL_SIZE = 150
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FONT_SIZE = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize the screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Policy Iteration Demo")

# Create the grid world
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Set obstacles (1 represents an obstacle)
obstacles = [(1, 2), (2, 2), (3, 2), (3, 3), (3, 4)]
for obs in obstacles:
    grid[obs] = 1

# Set goal state
goal_state = (4, 4)
grid[goal_state] = 2

# Initialize value function and policy
V = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.random.randint(0, 4, (GRID_SIZE, GRID_SIZE))

# Discount factor
gamma = 0.9

# Actions: up, right, down, left
actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
action_symbols = ["U", "R", "D", "L"]

def policy_evaluation():
    theta = 0.01
    while True:
        delta = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i, j] == 1 or (i, j) == goal_state:
                    continue
                v = V[i, j]
                a = actions[policy[i, j]]
                ni, nj = i + a[0], j + a[1]
                if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and grid[ni, nj] != 1:
                    V[i, j] = gamma * V[ni, nj]
                    if (ni, nj) == goal_state:
                        V[i, j] += 1  # Reward for reaching the goal
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break

def policy_improvement():
    policy_stable = True
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] == 1 or (i, j) == goal_state:
                continue
            old_action = policy[i, j]
            max_value = float('-inf')
            best_action = 0
            for a, action in enumerate(actions):
                ni, nj = i + action[0], j + action[1]
                if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and grid[ni, nj] != 1:
                    value = gamma * V[ni, nj]
                    if (ni, nj) == goal_state:
                        value += 1  # Reward for reaching the goal
                    if value > max_value:
                        max_value = value
                        best_action = a
            policy[i, j] = best_action
            if old_action != policy[i, j]:
                policy_stable = False
    return policy_stable

def policy_iteration():
    policy_evaluation()
    return policy_improvement()

def draw_grid():
    screen.fill(WHITE)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            
            if grid[i, j] == 1:  # Obstacle
                pygame.draw.rect(screen, RED, rect)
            elif (i, j) == goal_state:  # Goal
                pygame.draw.rect(screen, GREEN, rect)
            
            # Display value
            font = pygame.font.Font(None, FONT_SIZE)
            text = font.render(f"{V[i, j]:.2f}", True, BLUE)
            text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2 + 20))
            screen.blit(text, text_rect)

            # Display policy arrow
            if grid[i, j] != 1 and (i, j) != goal_state:
                arrow = action_symbols[policy[i, j]]
                arrow_text = font.render(arrow, True, BLACK)
                arrow_rect = arrow_text.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2 - 20))
                screen.blit(arrow_text, arrow_rect)

    # Display iteration count
    text = font.render(f"Iteration: {iteration}", True, BLACK)
    screen.blit(text, (10, 10))

    pygame.display.flip()

# Main game loop
running = True
iteration = 0
policy_stable = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not policy_stable:
                    policy_stable = policy_iteration()
                    iteration += 1
                    draw_grid()
                else:
                    print("Policy has converged!")

    # Initial draw
    if iteration == 0:
        draw_grid()

pygame.quit()
