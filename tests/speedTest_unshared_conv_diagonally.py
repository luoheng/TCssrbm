
import sys
import time
import unittest
import pdb

import numpy

import theano

from unshared_conv_diagonally import FilterActs
from unshared_conv_diagonally import WeightActs
from unshared_conv_diagonally import ImgActs


def rand(shp, dtype):
    return numpy.random.rand(*shp).astype(dtype)


class TestWeightActsSpeed(unittest.TestCase):

    # Global test variables (may be extended to include more tests)

    #Each item in ishape_list : (icount, icolors, irows, icols)
    ishape_list = [(64, 1, 98, 98)]

    #Each item in fshapes_list = (fmodules, filters_per_module, 
    #                             fcolors, frows, fcols)
    fshape_list = [(11, 32, 1, 11, 11)]

    # Each item in hshapes_list = (hcount, fmodules, filter_per_module, 
    #                              hrows, hcols)
    hshape_list = [(64, 11, 32, 8, 8)]

    module_stride = 1
    dtype = 'float64'
    nbTests = len(ishape_list)
    n_calls = 50

    # Utility functions
    def ishape(self, i):
        return self.ishape_list[i]

    def irows(self, i):
        return self.ishape_list[i][2]

    def icols(self, i):
        return self.ishape_list[i][3]

    def fshape(self, i):
        return self.fshape_list[i]
        
    def frows(self, i):
        return self.fshape_list[i][3]

    def fcols(self, i):
        return self.fshape_list[i][4]

    def hshape(self, i):
        return self.hshape_list[i]

    def setUp(self):
        self.op = WeightActs(module_stride=self.module_stride)
        
        self.s_images_list = [theano.shared(rand(ishape, self.dtype))
                              for ishape in self.ishape_list]
        self.s_hidacts_list = [theano.shared(rand(hshape, self.dtype))
                               for hshape in self.hshape_list]


    # Test Cases
    def testMainOpSpeed(self):
#        mode = theano.Mode(linker=theano.gof.vm.VM_Linker(
#            allow_gc=False,
#            use_cloop=True))
        for i in range(self.nbTests):

            # Generate theano functions to run the op in python and in C
            output = self.op(self.s_images_list[i], self.s_hidacts_list[i],
                             self.frows(i), self.fcols(i))

            pyFunction = theano.function([], output,
                                         mode=theano.Mode(linker='py'))

            cFunction = theano.function([], output,
                                        mode=theano.Mode(linker='c'))

            # Run the OP in python
            t0 = time.time()
            [pyFunction() for i in range(self.n_calls)]
            t1 = time.time()
            print "py", t1 - t0,

            # Run the OP in C and time it
            t0 = time.time()
            [cFunction() for i in range(self.n_calls)]
            t1 = time.time()
            print "c", t1 - t0

class TestWeightActsSpeedF32(TestWeightActsSpeed):
    dtype = 'float32'
    

class TestImgActsSpeed(unittest.TestCase):

    # Global test variables (may be extended to include more tests)

    #Each item in ishape_list : (icount, icolors, irows, icols)
    ishape_list = [(64, 1, 98, 98)]

    #Each item in fshapes_list = (fmodules, filters_per_module, 
    #                             fcolors, frows, fcols)
    fshape_list = [(11, 32, 1, 11, 11)]

    # Each item in hshapes_list = (hcount, fmodules, filter_per_module, 
    #                              hrows, hcols)
    hshape_list = [(64, 11, 32, 8, 8)]

    module_stride = 1
    dtype = 'float64'
    nbTests = len(ishape_list)
    n_calls = 50

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
        self.op = ImgActs(module_stride=self.module_stride, openmp=False)
        self.op_omp = ImgActs(module_stride=self.module_stride, openmp=True)

        self.s_filters_list = [theano.shared(rand(fshape, self.dtype))
                               for fshape in self.fshape_list]
        self.s_hidacts_list = [theano.shared(rand(hshape, self.dtype))
                               for hshape in self.hshape_list]

    # Test Cases
    def testMainOpSpeed(self):
#        mode = theano.Mode(linker=theano.gof.vm.VM_Linker(
#            allow_gc=False,
#            use_cloop=True))
        for i in range(self.nbTests):

            # Generate theano functions to run the op in python and in C
            output = self.op(self.s_filters_list[i], self.s_hidacts_list[i],
                             self.irows(i), self.icols(i))
            output_omp = self.op_omp(self.s_filters_list[i],
                                     self.s_hidacts_list[i],
                                     self.irows(i), self.icols(i))

            pyFunction = theano.function([], output,
                                         mode=theano.Mode(linker='py'))

            cFunction = theano.function([], output,
                                        mode=theano.Mode(linker='c'))

            cFunction2 = theano.function([], output_omp,
                                        mode=theano.Mode(linker='c'))
            # Run the OP in python
            t0 = time.time()
            [pyFunction() for i in range(self.n_calls)]
            t1 = time.time()
            py_t = t1 - t0
            print "py", py_t

            # Run the OP in C and time it
            t0 = time.time()
            [cFunction() for i in range(self.n_calls)]
            t1 = time.time()
            c_t = t1 - t0
            print "c", c_t, "speed up python", py_t / c_t

            # Run the Op in C with openmp
            if theano.config.openmp:
                t0 = time.time()
                [cFunction2() for i in range(self.n_calls)]
                t1 = time.time()
                c_t2 = t1 - t0
                print "omp c", c_t2, "speed up python", py_t / c_t2, "speed up c", c_t / c_t2


class TestImgActsSpeedF32(TestImgActsSpeed):
    dtype = 'float32'


class TestFiltersActsSpeedF64(unittest.TestCase):

    # Global test variables (may be extended to include more tests)

    #Each item in ishape_list : (icount, icolors, irows, icols)
    ishape_list = [(2, 1, 49, 49), (10, 1, 49, 49),
                   (10, 1, 98, 98), (10, 1, 98, 98),
                   (10, 1, 98, 98),
    ]

    #Each item in fshapes_list = (fmodules, filters_per_module,
    #                             fcolors, frows, fcols)
    fshape_list = [(5, 32, 1, 11, 11), (5, 32, 1, 11, 11),
                   (9, 32, 1, 9, 9),
                   (9, 32, 1, 10, 10), (9, 32, 1, 11, 11),

    ]
    
    ishape_list = [(64, 1, 98, 98)]
    fshape_list = [(11, 32, 1, 11, 11)]

    module_stride = 1
    dtype = 'float64'
    nbTests = len(ishape_list)
    n_calls = 50

    def setUp(self):
        self.op = FilterActs(module_stride=self.module_stride,
                                   openmp=False)
        self.op_omp = FilterActs(module_stride=self.module_stride,
                                    openmp=True)

        self.s_filters_list = [theano.shared(rand(fshape, self.dtype))
                               for fshape in self.fshape_list]
        self.s_images_list = [theano.shared(rand(ishape, self.dtype))
                               for ishape in self.ishape_list]

    # Test Cases
    def testMainOpSpeed(self):
        def do_time(output, mode=theano.Mode(linker='c')):
            f = theano.function([], output, mode=mode)
            t0 = time.time()
            [f() for i in range(self.n_calls)]
            t1 = time.time()
            return t1 - t0

        for i in range(self.nbTests):
            print "image shape", self.ishape_list[i]
            print "filter shape", self.fshape_list[i]
            # Generate theano functions to run the op in python and in C
            output = self.op(self.s_images_list[i], self.s_filters_list[i])
            output_omp = self.op_omp(self.s_images_list[i],
                                     self.s_filters_list[i])
            output_fcols = FilterActs(module_stride=self.module_stride,
                                      openmp=False,
                                      fcols=self.fshape_list[i][-1])(
                                          self.s_images_list[i],
                                          self.s_filters_list[i])
            output_fcols_omp = FilterActs(module_stride=self.module_stride,
                                      openmp=True,
                                      fcols=self.fshape_list[i][-1])(
                                          self.s_images_list[i],
                                          self.s_filters_list[i])
            output_frows_fcols = FilterActs(module_stride=self.module_stride,
                                      openmp=False,
                                      fcols=self.fshape_list[i][-1],
                                      frows=self.fshape_list[i][-2])(
                                          self.s_images_list[i],
                                          self.s_filters_list[i])
            output_frows_fcols_omp = FilterActs(module_stride=self.module_stride,
                                      openmp=True,
                                      fcols=self.fshape_list[i][-1],
                                      frows=self.fshape_list[i][-2])(
                                          self.s_images_list[i],
                                          self.s_filters_list[i])

            # Run the OP in python
            py_t = do_time(output, mode=theano.Mode(linker='py'))
            print "py", py_t

            # Run the OP in C
            c_t = do_time(output, mode=theano.Mode(linker='c|py'))
            print "c|py", c_t, "speed up", py_t / c_t

            # Run the OP in C with fcols
            c_t_fcols = do_time(output_fcols)
            print "c fcols", c_t_fcols, "speed up", py_t / c_t_fcols

            # Run the OP in C with fcols, frows
            c_t_frows_fcols = do_time(output_frows_fcols)
            print "c frows_fcols", c_t_frows_fcols, "speed up", py_t / c_t_frows_fcols

            # Run the Op in C with openmp
            if theano.config.openmp:
                c_omp_t = do_time(output_omp)
                print "omp c", c_omp_t, "speed up python", py_t / c_omp_t, "speed up c", c_t / c_omp_t

                c_omp_fcols_t = do_time(output_fcols_omp)
                print "omp c fcols", c_omp_fcols_t, "speed up python", py_t / c_omp_fcols_t, "speed up c fcols", c_t_fcols / c_omp_fcols_t

                c_omp_frows_fcols_t = do_time(output_frows_fcols_omp)
                print "omp c fcols", c_omp_frows_fcols_t, "speed up python", py_t / c_omp_frows_fcols_t, "speed up c frows_fcols", c_t_frows_fcols / c_omp_frows_fcols_t


class TestFiltersActsSpeedF32(TestFiltersActsSpeedF64):
    dtype = 'float32'
