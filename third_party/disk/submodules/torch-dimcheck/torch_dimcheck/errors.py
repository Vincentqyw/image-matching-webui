class ShapeError(AssertionError):
    pass


class SizeMismatchError(ShapeError):
    def __init__(self, dim, expected, found, tensor_name):
        self.dim = dim
        self.expected = expected
        self.found = found
        self.tensor_name = tensor_name

    def __str__(self):
        fmt = "Size mismatch on dimension {} of argument `{}` (found {}, expected {})"
        msg = fmt.format(self.dim, self.tensor_name, self.found, self.expected)
        return msg


class LabeledShapeError(ShapeError):
    def __init__(self, label, prev_binding, new_binding):
        self.label = label
        self.prev_binding = prev_binding
        self.new_binding = new_binding

    def __str__(self):
        fmt = ("Label `{}` already had dimension {} bound to it (based on tensor {} "
               "of shape {}), but it appears with dimension {} in tensor {}")
        msg = fmt.format(
            self.label, self.prev_binding.value, self.prev_binding.tensor_name,
            tuple(self.prev_binding.tensor_shape), self.new_binding.value,
            self.new_binding.tensor_name
        )
        return msg
