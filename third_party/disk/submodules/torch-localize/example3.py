import torch
from torch.nn import Linear, Sequential
import torch_localize

Linear = torch_localize.localized_module(Linear)

seq = Sequential(
    Linear(2, 4, name='linear1'),
    Linear(4, 3, name='linear2'),
    Linear(3, 7, name='linear3'),
    Linear(8, 2, name='linear4')
)

inp = torch.tensor([1., 0.])

print(seq(inp))
