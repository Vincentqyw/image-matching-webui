# torch-localize
Decorators for better tracebacks in complex PyTorch models.

# Problem description
If we make an error writing simple models, in which we manually call each module,
for [instance](example1.py)

```python
import torch
from torch.nn import Module, Linear

class MyModule(Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.lin1 = Linear(2, 4)
        self.lin2 = Linear(5, 3)

    def forward(self, inp):
        y = self.lin1(inp)
        y = self.lin2(y)

        return y
```

we get stack traces which show the exact location of the error we made. If we execute

```python
inp = torch.tensor([1., 0.])
mod = MyModule()

print(mod(inp))
```

we will be told that the offending line is `y = self.lin2(y)`

```
  File "example3.py", line 19, in <module>
    print(mod(inp))
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "example3.py", line 12, in forward
    y = self.lin2(y)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 55, in forward
    return F.linear(input, self.weight, self.bias)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/functional.py", line 1026, in linear
    output = input.matmul(weight.t())
RuntimeError: size mismatch, m1: [1 x 4], m2: [5 x 3] at /opt/conda/conda-bld/pytorch-cpu_1532576596369/work/aten/src/TH/generic/THTensorMath.cpp:2070
```

This unfortunately doesn't apply anymore when using loops and other forms
of flow control in our model. For [example](example2.py)

```python
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
```

results in the following traceback:

```
Traceback (most recent call last):
  File "example1.py", line 13, in <module>
    print(seq(inp))
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/container.py", line 91, in forward
    input = module(input)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 55, in forward
    return F.linear(input, self.weight, self.bias)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/functional.py", line 1026, in linear
    output = input.matmul(weight.t())
RuntimeError: size mismatch, m1: [1 x 7], m2: [8 x 2] at /opt/conda/conda-bld/pytorch-cpu_1532576596369/work/aten/src/TH/generic/THTensorMath.cpp:2070
```

from which we can only find that the error is in one of the linear layers, but
we are not told which one (in this toy example it's easy enough to figure out
by looking at the sizes mentioned in the `RuntimeError`, but it's not always the 
case). This repository introduces a decorator called `localized_module` which
decorates a module, adding an optional `name` parameter to its `__init__`,
automatically assigning it to `.name` attribute of the module and and wraps its
`forward` method to include this name in traceback when an exception happens.
Now, our code looks like [this](example3.py):

```python
import torch
from torch.nn import Linear, Sequential
import torch_localize

# decorate Linear to allow specifying names
Linear = torch_localize.localized_module(Linear)

seq = Sequential(
    Linear(2, 4, name='linear1'),
    Linear(4, 3, name='linear2'),
    Linear(3, 7, name='linear3'),
    Linear(8, 2, name='linear4')
)

inp = torch.tensor([1., 0.])

print(seq(inp))
```

and results in the following traceback:

```
Traceback (most recent call last):
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch_localize-0.0.1-py3.6.egg/torch_localize/localize.py", line 14, in wrapped
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 55, in forward
    return F.linear(input, self.weight, self.bias)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/functional.py", line 1026, in linear
    output = input.matmul(weight.t())
RuntimeError: size mismatch, m1: [1 x 7], m2: [8 x 2] at /opt/conda/conda-bld/pytorch-cpu_1532576596369/work/aten/src/TH/generic/THTensorMath.cpp:2070

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "example2.py", line 16, in <module>
    print(seq(inp))
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/container.py", line 91, in forward
    input = module(input)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/jatentaki/anaconda3/lib/python3.6/site-packages/torch_localize-0.0.1-py3.6.egg/torch_localize/localize.py", line 19, in wrapped
torch_localize.localize.LocalizedException: Exception in linear4
```

Where we are told explicitly that the exception occured in module named `linear4`.
While the examples given here are toy, I found this decorator very useful for
models which make use of `nn.ModuleList` and `nn.ModuleDict`. An example is when
writing generic network constructors for Unets with variable depth
and numbers of feature maps at each layer.

# API
This module exports two decorators: `localized` and `localized_module` as well as `LocalizedException`.
## `localized`
`localized` is a method wrapper which runs the original method and, in case of an exception, checks if `self` has attribute `name`. If so, it raises `LocalizedException` from the original exception, including `name` in the message. If `name` is not present among attributes, the message will default to `<unnamed ClassName>`. It is provided in case somebody wants to localize exceptions in other methods than just `forward` (which is being decorated by default, when using `localized_module`).
## `localized_module`
`localized_module` modifies class `__init__` to include `name=None` in its argument list and assigns the value to objects `name` attribute, as well as wraps the `forward` method with `localized`. `localized_module` performs a bunch of sanity checks to make sure it's not overwriting existent parameter/attribute names, and should be fool proof enough to detect most attempts to misuse it. One way to cause trouble I can think of is when wrapping a class which depends on catching a `name` parameter in its `**kwargs`, since the `name` parameter is being deleted from `**kwargs` before passing it to class `__init__`. I expect this corner case to be exceedingly rare in real code.
## `LocalizedException`
This class is provided in case somebody wants to extent the functionality of this library
