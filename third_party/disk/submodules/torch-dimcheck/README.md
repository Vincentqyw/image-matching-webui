# torch-dimcheck
Dimensionality annotations for tensor parameters and return values.

## Installation
1. Clone the repository
2. Run `python setup.py install`

## Usage
This module uses the python type annotations to provide run-time argument size checking for PyTorch tensors, allowing for writing
* Self-documenting code (in a way which doesn't silently become outdated)
* Fail-fast code, where the error points to the first location where a contract was violated

the only user-facing part of torch_dimcheck is the `dimchecked` function decorator:

```python
import torch
from torch_dimcheck import dimchecked

@dimchecked
def matmul(a: ['X', 'Y'], ['Y', 'Z']) -> ['X', 'Z']:
  return torch.matmul(a, b)

a = torch.randn(3, 4)
b = torch.randn(4, 2)

c = matmul(a, b) # works
c = matmul(b, a) # throws at function call level
```

### In-depth description and advanced features
Each function parameter and output value can be annotated with a `list` where each element is either `str`, `int` or [`...`](https://docs.python.org/3/library/constants.html#Ellipsis). We refer to the elements of the list as *labels* and say that
1. The tensor will be required to have as many dimensions as the list has labels. 
2. `int` labels require the tensor dimension to have size equal to that value (i.e. `f(a: [1, 4])` will accept only tensors of shape `[1, 4]`)
3. `str` labels create a unique *dynamic* label, which can have any size but must be consistent across the whole signature. This means that in `add(a: ['A'], b: ['A'])` the tensors must be 1-dimensional and of equal size
4. Ellipsis `...` is a special value which can stand for *any* amount of dimensions, thus being a way to violate rule 1. There can be at most one `...` per tensor annotation (otherwise the notation would be ambiguous). For example, `g(a: ['A', ..., 'B'], b: ['A', ..., 'B'])` means that `a` and `b` can have an arbitrary amount of dimensions as long as the first and last ones agree in size.
5. Argument annotations other than `list`s are ignored, which means that one can still use regular type hints alongside `@dimchecked`.

Additionally, function outputs are annotated as a `tuple` of `list`s, with each `list` referring to one function output.

```python
@dimchecked
def matmul_two_ways(a: ['X', 'Y'], b: ['Y', 'Z']) -> (['X', 'Z'], ['Z', 'X']):
  ab = torch.matmul(a, b)
  ba = torch.matmul(b, a)
  return ab, ba
```

In this context `...` has a special meaning and can replace a `list`, meaning that this output will not be checked: this is useful if only part of the function outputs are tensors.
```python
@dimchecked
def load_ith_image(i) -> (['H', 'W', 3], ...):
  path = find_ith_path(i)
  return load_image(path), path
```

Finally, if there is only a single tensor as an output, the outer `tuple` can be skipped:

```python
@dimchecked
def f() -> ['X', 'Y']:
  pass
 
# is equivalent to

@dimchecked
def f() -> (['X', 'Y'], ):
  pass
```
