import torch, functools, inspect

from .errors import ShapeError, SizeMismatchError, LabeledShapeError

class Binding:
    def __init__(self, label, value, tensor_name, tensor_shape):
        self.label = label
        self.value = value
        self.tensor_name = tensor_name
        self.tensor_shape = tensor_shape

class ShapeChecker:
    def __init__(self):
        self.d = dict()

    def update(self, other):
        if isinstance(other, ShapeChecker):
            other = other.d

        for label in other.keys():
            if label in self.d:
                binding = self.d[label]
                new_binding = other[label]

                if not binding.value == new_binding.value:
                    raise LabeledShapeError(label, binding, new_binding)
            else:
                self.d[label] = other[label]

    def check(self, tensor, annotation, name=None):
        bindings = get_bindings(tensor, annotation, tensor_name=name)
        self.update(bindings)


def get_bindings(tensor, annotation, tensor_name=None):
    if not isinstance(tensor, torch.Tensor):
        fmt = "Expected argument `{}` to be an instance of torch.Tensor, found {} instead"
        msg = fmt.format(tensor_name, type(tensor))
        raise ValueError(msg)

    n_ellipsis = annotation.count(...)
    if n_ellipsis > 1:
        # TODO: check this condition earlier
        raise ValueError("Only one ellipsis can be used per annotation")

    if len(annotation) != len(tensor.shape) and n_ellipsis == 0:
        # no ellipsis, dimensionality mismatch
        fmt = "Annotation {} differs in size from tensor shape {} ({} vs {})"
        msg = fmt.format(annotation, tuple(tensor.shape), len(annotation), len(tensor.shape))
        raise ShapeError(msg)

    bindings = ShapeChecker()
    # check if dimensions match, one by one
    for i, (dim, anno) in enumerate(zip(tensor.shape, annotation)):
        if isinstance(anno, str):
            # named wildcard, add to dict
            bindings.update({anno: Binding(anno, dim, tensor_name, tensor.shape)})
        elif anno == ...:
            # ellipsis - done checking from the front, skip to checking in reverse
            break
        elif isinstance(anno, int) and anno != dim:
            if anno == -1:
                # anonymous wildcard dimension, continue
                continue
            else:
                raise SizeMismatchError(i, anno, dim, tensor_name)

    if n_ellipsis == 0:
        # no ellipsis - we don't have to go in reverse
        return bindings

    # there was an ellipsis, we have to check in reverse
    for i, (dim, anno) in enumerate(zip(tensor.shape[::-1], annotation[::-1])):
        if isinstance(anno, str):
            # named wildcard, add to dict
            bindings.update({anno: Binding(anno, dim, tensor_name, tensor.shape)})
        elif anno == ...:
            # ellipsis - done checking from the back, return
            return bindings
        elif isinstance(anno, int) and anno != dim:
            if anno == -1:
                # anonymous wildcard dimension, continue
                continue
            else:
                raise SizeMismatchError(len(tensor.shape) - i - 1, anno, dim, tensor_name)

    raise AssertionError("Arrived at the end of procedure")


def dimchecked(func):
    sig = inspect.signature(func)

    checked_parameters = dict()
    for i, parameter in enumerate(sig.parameters.values()):
        if isinstance(parameter.annotation, list):
            checked_parameters[i] = parameter

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # check input
        shape_bindings = ShapeChecker()
        for i, arg in enumerate(args):
            if i in checked_parameters:
                param = checked_parameters[i]
                shapes = get_bindings(
                    arg, param.annotation, tensor_name=param.name
                )
                shape_bindings.update(shapes)

        result = func(*args, **kwargs)

        if isinstance(sig.return_annotation, list):
            # single tensor output like f() -> [3, 6]
            shapes = get_bindings(
                result, sig.return_annotation, tensor_name='<return value>'
            )
            shape_bindings.update(shapes)
        elif isinstance(sig.return_annotation, tuple):
            # tuple output like f() -> ([3, 6], ..., [6, 5])
            for i, anno in enumerate(sig.return_annotation):
                if anno == ...:
                    # skip
                    continue

                shapes = get_bindings(
                    result[i], anno, tensor_name='<return value {}>'.format(i)
                )
                shape_bindings.update(shapes)

        return result

    return wrapped
