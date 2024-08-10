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
pygame.display.set_caption("Value Iteration Demo")

# Create the grid world
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Set obstacles (1 represents an obstacle)
obstacles = [(1, 2), (2, 2), (3, 2), (3, 3), (3, 4)]
for obs in obstacles:
    grid[obs] = 1

# Set goal state
goal_state = (4, 4)
grid[goal_state] = 2

# Initialize value function
V = np.zeros((GRID_SIZE, GRID_SIZE))

# Discount factor
gamma = 0.9

# Actions: up, right, down, left
actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def value_iteration(iterations=1):
    for _ in range(iterations):
        new_V = np.zeros_like(V)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i, j] == 1:  # Skip obstacles
                    continue
                if (i, j) == goal_state:
                    new_V[i, j] = 1  # Reward for reaching the goal
                    continue
                max_value = float('-inf')
                for action in actions:
                    ni, nj = i + action[0], j + action[1]
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and grid[ni, nj] != 1:
                        max_value = max(max_value, V[ni, nj])
                new_V[i, j] = gamma * max_value
        V[:] = new_V

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
            text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2))
            screen.blit(text, text_rect)

    # Display iteration count
    font = pygame.font.Font(None, FONT_SIZE)
    text = font.render(f"Iteration: {iteration}", True, BLACK)
    screen.blit(text, (10, 10))

    pygame.display.flip()

# Main game loop
running = True
iteration = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                value_iteration(1)
                iteration += 1
                draw_grid()

    # Initial draw
    if iteration == 0:
        draw_grid()

pygame.quit()