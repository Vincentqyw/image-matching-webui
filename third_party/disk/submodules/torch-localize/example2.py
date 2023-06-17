import torch
from torch.nn import Linear, Sequential

seq = Sequential(
    Linear(2, 4),
    Linear(4, 3),
    Linear(3, 7),
    Linear(8, 2)
)

inp = torch.tensor([1., 0.])

print(seq(inp))
