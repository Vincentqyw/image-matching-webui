import functools, inspect
from torch.nn import Module


class LocalizedException(Exception):
    pass


def default_name(entity):
    return '<unnamed {}>'.format(type(entity).__name__)


def localized(method):
    ''' Adds module name to traceback '''
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            name = getattr(self, 'name', None)
            if name is None:
                name = default_name(self)
            raise LocalizedException('Exception in ' + name) from e

    return wrapped


def localized_module(cls):
    ''' 
    Decorates a Module class to register .name parameter in its __init__
    and decorates its forward(...) with @localized
    '''

    if not issubclass(cls, Module):
        raise TypeError("Can only localize torch.nn.Module subclasses")

    cls_init = cls.__init__
    signature = inspect.signature(cls_init)
    if 'name' in signature.parameters:
        fmt = "Class {} __init__ already has 'name' in signature"
        msg = fmt.format(cls.__name__)
        raise ValueError(msg)
    
    @functools.wraps(cls_init)
    def new_init(self, *args, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name']
            # delete it so it doesn't propagate to the wrapped class' __init__
            del kwargs['name']
        else:
            name = default_name(self)

        cls_init(self, *args, **kwargs)
        if hasattr(self, 'name'):
            fmt = "Instance {} of class {} already has .name set via the base __init__"
            msg = fmt.format(repr(self), cls.__name__)
            raise RuntimeError(msg)

        self.name = name

    cls.__init__ = new_init
    cls.forward = localized(cls.forward)

    return cls

if __name__ == '__main__':
    import unittest

    class Tests(unittest.TestCase):
        def test_correct_named(self):
            @localized_module
            class Mod(Module):
                def __init__(self):
                    super(Mod, self).__init__()

                def forward(self, input):
                    raise AssertionError("I have failed")

            m = Mod(name='Slim Shady')
            localization = 'Exception in Slim Shady'

            with self.assertRaisesRegex(LocalizedException, localization):
                m.forward('Hi')


        def test_correct_unnamed(self):
            @localized_module
            class Mod(Module):
                def __init__(self):
                    super(Mod, self).__init__()

                def forward(self, input):
                    raise AssertionError("I have failed")

            m = Mod()
            localization = 'Exception in <unnamed {}>'.format(Mod.__name__)

            with self.assertRaisesRegex(LocalizedException, localization):
                m.forward('Hi')


        def test_not_module(self):
            with self.assertRaises(TypeError):
                @localized_module
                class NotAModule:
                    def __init__(self):
                        self.property = 'Hi!'


        def test_takes_name_arg_in_init(self):
            with self.assertRaises(ValueError):
                @localized_module
                class TakesAName(Module):
                    def __init__(self, name):
                        super(TakesAName, self).__init__()


        def test_takes_name_kwarg_in_init(self):
            with self.assertRaises(ValueError):
                @localized_module
                class TakesAName(Module):
                    def __init__(self, name='Slim Shady'):
                        super(TakesAName, self).__init__()


        def test_has_name_as_property(self):
            @localized_module
            class HasAName(Module):
                def __init__(self):
                    super(HasAName, self).__init__()
                    self.name = 'Slim Shady'

            with self.assertRaises(RuntimeError):
                HasAName()

    unittest.main()
