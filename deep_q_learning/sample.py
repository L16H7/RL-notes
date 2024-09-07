import torch
from neural_network import DQN

model = DQN(8, 4)

random_state = torch.tensor([1, 1, 0, 0, -5, -2, 9, 10], dtype=torch.float32)

output = model.forward(random_state)

print(output)
