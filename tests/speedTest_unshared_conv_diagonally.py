
import sys
import unittest
import pdb

import numpy

import theano

from unshared_conv_diagonally import FilterActs
from unshared_conv_diagonally import WeightActs
from unshared_conv_diagonally import ImgActs

def rand(shp, dtype):
    return numpy.random.rand(*shp).astype(dtype)

class TestImgActsSpeedF64(unittest.TestCase):

    # Global test variables (may be extended to include more tests)

    #Each item in ishape_list : (icount, icolors, irows, icols)
    ishape_list = [(10, 1, 98, 98)]
    
    #Each item in fshapes_list = (fmodules, filters_per_module, 
    #                             fcolors, frows, fcols)
    fshape_list = [(11, 32, 1, 11, 11)]

    # Each item in hshapes_list = (hcount, fmodules, filter_per_module, 
    #                              hrows, hcols)
    hshape_list = [(10, 11, 32, 8, 8)]

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

    def testMainOpSpeed(self):
        
        for i in range(self.nbTests):
                                                 
            # Generate theano functions to run the op in python and in C
            output = self.op(self.s_filters_list[i], self.s_hidacts_list[i],
                             self.irows(i), self.icols(i))
            
            pyFunction = theano.function([],output,
                                         mode=theano.Mode(linker='py'))
            cFunction = theano.function([],output,
                                        mode=theano.Mode(linker='c'))
            
             
            # Run the OP in python and (TODO)time it
            for noRun in range(10):
                pyResult = pyFunction()
                                
            # Run the OP in C and (TODO)time it
            for noRun in range(10):
                cResult = cFunction()
                    

class TestImgActsSpeedF32(unittest.TestCase):
    dtype = 'float32'
