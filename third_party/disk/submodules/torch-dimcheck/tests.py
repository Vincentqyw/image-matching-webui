import unittest, torch
from torch_dimcheck import dimchecked
from torch_dimcheck.errors import ShapeError

class ShapeCheckedTests(unittest.TestCase):
    def test_wrap_no_anno(self):
        def f(t1, t2): # t1: [3, 5], t2: [5, 3] -> [3]
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())

    def test_wrap_correct(self):
        def f(t1: [3, 5], t2: [5, 3]) -> [3]:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())

    def test_fails_wrong_parameter(self):
        def f(t1: [3, 3], t2: [5, 3]) -> [3]:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        msg = "Size mismatch on dimension 1 of argument `t1` (found 5, expected 3)"
        with self.assertRaises(ShapeError) as ex:
            dimchecked(f)(t1, t2)
        self.assertEqual(str(ex.exception), msg)

    def test_fails_backward_ellipsis(self):
        def f(t1: [3, ..., 2], t2: [5, ..., 3]):
            pass
             
        t1 = torch.randn(3, 3, 5)
        t2 = torch.randn(5, 3, 3)

        msg = "Size mismatch on dimension 2 of argument `t1` (found 5, expected 2)"
        with self.assertRaises(ShapeError) as ex:
            dimchecked(f)(t1, t2)
        self.assertEqual(str(ex.exception), msg)

    def test_fails_backward_ellipsis_wildcard(self):
        def f(t1: [3, ..., 'a'], t2: [5, ..., 'a']):
            pass
             
        t1 = torch.randn(3, 3, 5)
        t2 = torch.randn(5, 3, 3)

        msg = ("Label `a` already had dimension 5 bound to it "
               "(based on tensor t1 of shape (3, 3, 5)), but it "
               "appears with dimension 3 in tensor t2")
        with self.assertRaises(ShapeError) as ex:
            dimchecked(f)(t1, t2)
        self.assertEqual(str(ex.exception), msg)

    def test_fails_backward_just_ellipsis(self):
        def f(t1: [..., 2], t2: [..., 2]):
            pass
             
        t1 = torch.randn(3, 3, 3, 2)
        t2 = torch.randn(5, 3, 1, 5)

        msg = "Size mismatch on dimension 3 of argument `t2` (found 5, expected 2)"
        with self.assertRaises(ShapeError) as ex:
            dimchecked(f)(t1, t2)
        self.assertEqual(str(ex.exception), msg)

    def test_fails_wrong_return(self):
        def f(t1: [3, 5], t2: [5, 3]) -> [5]:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        msg = ("Size mismatch on dimension 0 of argument "
               "`<return value>` (found 3, expected 5)")
        with self.assertRaises(ShapeError) as ex:
            dimchecked(f)(t1, t2)
        self.assertEqual(str(ex.exception), msg)

    def test_fails_parameter_label_mismatch(self):
        def f(t1: [3, 'a'], t2: ['a', 3]) -> [3]:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 4)
        t2 = torch.randn(5, 3)

        with self.assertRaises(ShapeError):
            dimchecked(f)(t1, t2)

    def test_fails_return_label_mismatch(self):
        def f(t1: [5, 'a'], t2: ['a', 5]) -> ['a']:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        with self.assertRaises(ShapeError):
            dimchecked(f)(t1, t2)

    def test_succeeds_ellipsis(self):
        def f(t1: [5, ..., 'a'], t2: ['a', ..., 5]) -> ['a']:
            return (t1.transpose(0, 3) * t2).sum(dim=(1, 2, 3))
             
        t1 = torch.randn(5, 1, 2, 3)
        t2 = torch.randn(3, 1, 2, 5)

        self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())

unittest.main()
