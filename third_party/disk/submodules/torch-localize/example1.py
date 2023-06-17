import torch
from torch.nn import Module, Linear, Sequential

class MyModule(Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.lin1 = Linear(2, 4)
        self.lin2 = Linear(5, 3)

    def forward(self, inp):
        y = self.lin1(inp)
        y = self.lin2(y)

        return y

inp = torch.tensor([1., 0.])
mod = MyModule()

print(mod(inp))
