import sys
import unittest
import pdb

import numpy

import theano
from theano.tests.unittest_tools import verify_grad

from unshared_conv_diagonally import FilterActs
from unshared_conv_diagonally import WeightActs
from unshared_conv_diagonally import ImgActs


def rand(shp, dtype):
#    return numpy.ones(shp, dtype=dtype)
#    return numpy.arange(numpy.prod(shp)).reshape(shp).astype(dtype)
    return numpy.random.rand(*shp).astype(dtype)


def assert_linear(f, pt, mode=None):
    t = theano.tensor.scalar(dtype=pt.dtype)
    ptlike = theano.shared(rand(
        pt.get_value(borrow=True).shape,
        dtype=pt.dtype))
    out = f(pt)
    out2 = f(pt * t)
    out3 = f(ptlike) + out
    out4 = f(pt + ptlike)

    f = theano.function([t], [out * t, out2, out3, out4],
            allow_input_downcast=True,
            mode=mode)
    outval, out2val, out3val, out4val = f(3.6)
    assert numpy.allclose(outval, out2val)
    assert numpy.allclose(out3val, out4val)


class TestFilterActs(unittest.TestCase):

    # Global test variables (may be extended to include more tests)

    #Each item in ishape_list : (icount, icolors, irows, icols)
    ishape_list = [(1, 2, 6, 6), (1, 2, 6, 6),
                   (2, 3, 24, 24), (2, 3, 20, 20),
                   (2, 3, 49, 49), (20, 1, 98, 98)]

    #Each item in fshapes_list = (fmodules, filters_per_module,
    #                             fcolors, frows, fcols)
    fshape_list = [(1, 1, 2, 3, 3), (1, 1, 2, 3, 3),
                   (1, 1, 3, 6, 6), (3, 2, 3, 6, 6),
                   (5, 32, 3, 11, 11), (11, 32, 1, 11, 11)]

    # Each item in hshapes_list = (hcount, fmodules, filter_per_module,
    #                              hrows, hcols)
    hshape_list = [(1, 1, 1, 2, 2), (1, 1, 1, 2, 2),
                   (2, 1, 1, 4, 4), (2, 3, 2, 3, 3),
                   (2, 5, 32, 4, 4), (20, 11, 32, 8, 8)]

    module_stride = 1
    dtype = 'float64'
    mode = theano.compile.get_default_mode()
    nbTests = len(ishape_list)

    # Utility functions
    def ishape(self, i):
        return self.ishape_list[i]

    def fshape(self, i):
        return self.fshape_list[i]

    def hshape(self, i):
        return self.hshape_list[i]

    def function(self, inputs, outputs):
        return theano.function(inputs, outputs, mode=self.mode)

    def setUp(self):
        self.op = FilterActs(self.module_stride)

        for i in range(self.nbTests):
            self.s_images_list = [theano.shared(rand(ishape, self.dtype))
                                  for ishape in self.ishape_list]
            self.s_filters_list = [theano.shared(rand(fshape, self.dtype))
                                   for fshape in self.fshape_list]

    # Test cases
    def test_type(self):
        for i in range(self.nbTests):

            out = self.op(self.s_images_list[i], self.s_filters_list[i])
            assert out.dtype == self.dtype
            assert out.ndim == 5

            f = self.function([], out)
            outval = f()
            assert outval.shape == self.hshape(i)
            assert outval.dtype == self.s_images_list[i].get_value(
                borrow=True).dtype

    def test_linearity_images(self):
        for i in range(self.nbTests):
            assert_linear(
                    lambda imgs: self.op(imgs, self.s_filters_list[i]),
                    self.s_images_list[i],
                    mode=self.mode)

    def test_linearity_filters(self):
        for i in range(self.nbTests):
            assert_linear(
                    lambda fts: self.op(self.s_images_list[i], fts),
                    self.s_filters_list[i],
                    mode=self.mode)

    def test_shape(self):
        for i in range(self.nbTests):
            out = self.op(self.s_images_list[i], self.s_filters_list[i])
            f = self.function([], out)
            outval = f()

            assert outval.shape == self.hshape(i)

    def test_grad_left(self):
        for i in range(self.nbTests - 2):

            # test only the left so that the right can be a shared variable,
            # (for tests on the GPU)
            def left_op(imgs):
                return self.op(imgs, self.s_filters_list[i])
            try:
                verify_grad(left_op, [self.s_images_list[i].get_value()],
                            mode=self.mode, eps=6e-4)
            except verify_grad.E_grad, e:
                raise
                print e.num_grad.gf
                print e.analytic_grad
                raise

    def test_grad_right(self):
        for i in range(self.nbTests - 2):

            # test only the right so that the left can be a shared variable,
            # (for tests on the GPU)
            def right_op(filters):
                return self.op(self.s_images_list[i], filters)

            try:
                verify_grad(right_op, [self.s_filters_list[i].get_value()],
                            mode=self.mode, eps=3e-4)#rel_tol=0.0006)
            except verify_grad.E_grad, e:
                raise
                print e.num_grad.gf
                print e.analytic_grad
                raise

    def test_dtype_mismatch(self):
        for i in range(self.nbTests):

            self.assertRaises(TypeError,
                    self.op,
                    theano.tensor.cast(self.s_images_list[i], 'float32'),
                    theano.tensor.cast(self.s_filters_list[i], 'float64'))
            self.assertRaises(TypeError,
                    self.op,
                    theano.tensor.cast(self.s_images_list[i], 'float64'),
                    theano.tensor.cast(self.s_filters_list[i], 'float32'))

    def test_op_eq(self):
        assert FilterActs(1) == FilterActs(1)
        assert not (FilterActs(1) != FilterActs(1))
        assert (FilterActs(2) != FilterActs(1))
        assert FilterActs(1) != None


class TestFilterActsF32(TestFilterActs):
    dtype = 'float32'


class TestWeightActs(unittest.TestCase):

    # Global test variables (may be extended to include more tests)

    #Each item in ishape_list : (icount, icolors, irows, icols)
    #ishape_list = [(1, 1, 98, 98),(2, 3, 24, 24)]
    ishape_list = [(10, 1, 98, 98)]

    #Each item in fshapes_list = (fmodules, filters_per_module,
    #                             fcolors, frows, fcols)

    #fshape_list = [(11, 32, 1, 11, 11),(1, 1, 3, 6, 6)]
    fshape_list = [(11, 32, 1, 11, 11)]

    # Each item in hshapes_list = (hcount, fmodules, filter_per_module,
    #                              hrows, hcols)
    #hshape_list = [(1, 11, 32, 8, 8),(2, 1, 1, 6, 6 )]
    hshape_list = [(10, 11, 32, 8, 8)]

    module_stride = 1
    dtype = 'float64'
    nbTests = len(ishape_list)

    # Utility functions
    def ishape(self, i):
        return self.ishape_list[i]

    def fshape(self, i):
        return self.fshape_list[i]

    def frows(self, i):
        return self.fshape_list[i][3]

    def fcols(self, i):
        return self.fshape_list[i][4]

    def hshape(self, i):
        return self.hshape_list[i]

    def setUp(self):
        self.op = WeightActs(self.module_stride)

        for i in range(self.nbTests):
            self.s_images_list = [theano.shared(rand(ishape, self.dtype))
                                  for ishape in self.ishape_list]
            self.s_hidacts_list = [theano.shared(rand(hshape, self.dtype))
                                   for hshape in self.hshape_list]

    # Test cases
    def test_type(self):
        for i in range(self.nbTests):

            out = self.op(self.s_images_list[i], self.s_hidacts_list[i],
                          self.frows(i), self.fcols(i))
            assert out.dtype == self.dtype
            assert out.ndim == 5
            f = theano.function([], out)
            outval = f()
            assert outval.shape == self.fshape(i)
            assert outval.dtype == self.dtype

    def test_linearity_images(self):
        for i in range(self.nbTests):

            def f(images):
                return self.op(images, self.s_hidacts_list[i],
                               self.frows(i), self.fcols(i))
            assert_linear(f, self.s_images_list[i])

    def test_linearity_hidacts(self):
        for i in range(self.nbTests):

            def f(hidacts):
                return self.op(self.s_images_list[i], hidacts,
                               self.frows(i), self.fcols(i))
            assert_linear(f, self.s_hidacts_list[i])

    def test_grad(self):
        for i in range(self.nbTests):

            def op2(imgs, hids):
                return self.op(imgs, hids, self.frows(i), self.fcols(i))
            try:
                verify_grad(op2,
                            [self.s_images_list[i].get_value(),
                             self.s_hidacts_list[i].get_value()])
            except verify_grad.E_grad, e:
                print e.num_grad.gf
                print e.analytic_grad
                raise

    def test_dtype_mismatch(self):
        for i in range(self.nbTests):

            self.assertRaises(TypeError,
                    self.op,
                    theano.tensor.cast(self.s_images_list[i], 'float32'),
                    theano.tensor.cast(self.s_hidacts_list[i], 'float64'),
                    self.frows, self.fcols)
            self.assertRaises(TypeError,
                    self.op,
                    theano.tensor.cast(self.s_images_list[i], 'float64'),
                    theano.tensor.cast(self.s_hidacts_list[i], 'float32'),
                    self.frows, self.fcols)


class TestWeightActsF32(TestWeightActs):
    dtype = 'float32'


class TestImgActs(unittest.TestCase):

    # Global test variables (may be extended to include more tests)

    #Each item in ishape_list : (icount, icolors, irows, icols)
    ishape_list = [(10, 1, 98, 98)]

    #Each item in fshapes_list = (fmodules, filters_per_module,
    #                             fcolors, frows, fcols)
    fshape_list = [(11, 32, 1, 11, 11)]

    # Each item in hshapes_list = (hcount, fmodules, filter_per_module,
    #                              hrows, hcols)
    hshape_list = [(10, 11, 32, 9, 9)]

    
    module_stride = 1
    dtype = 'float64'
    nbTests = len(ishape_list)

    # Utility functions
    def ishape(self, i):
        return self.ishape_list[i]

    def irows(self, i):
        return self.ishape_list[i][2]

    def icols(self, i):
        return self.ishape_list[i][3]

    def fshape(self, i):
        return self.fshape_list[i]

    def hshape(self, i):
        return self.hshape_list[i]

    def setUp(self):
        self.op = ImgActs(module_stride=self.module_stride)

        self.s_filters_list = [theano.shared(rand(fshape, self.dtype))
                               for fshape in self.fshape_list]
        self.s_hidacts_list = [theano.shared(rand(hshape, self.dtype))
                               for hshape in self.hshape_list]

    # Test Cases
    def test_type(self):

        for i in range(self.nbTests):

            out = self.op(self.s_filters_list[i], self.s_hidacts_list[i],
                          self.irows(i), self.icols(i))

            assert out.dtype == self.dtype
            assert out.ndim == 4
            f = theano.function([], out)
            outval = f()
            assert outval.shape == self.ishape(i)
            assert outval.dtype == self.dtype

    def test_linearity_filters(self):
        for i in range(self.nbTests):

            def f(filts):
                return self.op(filts, self.s_hidacts_list[i],
                               self.irows(i), self.icols(i))
            assert_linear(f, self.s_filters_list[i])

    def test_linearity_hidacts(self):
        for i in range(self.nbTests):

            def f(hidacts):
                return self.op(self.s_filters_list[i], hidacts,
                               self.irows(i), self.icols(i))

            assert_linear(f, self.s_hidacts_list[i])

    def test_grad(self):
        for i in range(self.nbTests):

            def op2(imgs, hids):
                return self.op(imgs, hids, self.irows(i), self.icols(i))
            try:
                verify_grad(op2,
                            [self.s_filters_list[i].get_value(),
                             self.s_hidacts_list[i].get_value()])
            except verify_grad.E_grad, e:
                print e.num_grad.gf
                print e.analytic_grad
                raise

    def test_dtype_mismatch(self):
        for i in range(self.nbTests):

            self.assertRaises(TypeError,
                    self.op,
                    theano.tensor.cast(self.s_filters_list[i], 'float32'),
                    theano.tensor.cast(self.s_hidacts_list[i], 'float64'),
                    self.irows(i), self.icols(i))
            self.assertRaises(TypeError,
                    self.op,
                    theano.tensor.cast(self.s_filters_list[i], 'float64'),
                    theano.tensor.cast(self.s_hidacts_list[i], 'float32'),
                    self.irows(i), self.icols(i))


class TestImgActsF32(TestImgActs):
    dtype = 'float32'
