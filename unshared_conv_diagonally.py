"""
XXX
"""

import pdb
import numpy
import theano
import StringIO

from theano.tensor import blas
from theano import gof, tensor, scalar

def any_symbolic(*args):
    """
    Return True iff any a in `args` is a theano Variable
    """
    for a in args:
        if isinstance(a, theano.Variable):
            return True
    return False

def not_symbolic(*args):
    return not any_symbolic(*args)


class Base(theano.Op):
    def __init__(self,
            module_stride=1,
            openmp=None
            ):
        self.module_stride = module_stride
        if openmp is None:
            openmp = theano.config.openmp
        self.openmp = openmp

    def _attributes(self):
        return (
                self.module_stride,
            self.openmp
                )

    def __eq__(self, other):
        return (type(self) == type(other)
                and self._attributes() == other._attributes())

    def __hash__(self):
        return hash((type(self), self._attributes()))

    def __str__(self):
        return '%s{module_stride=%i,openmp=%d}' % (
                self.__class__.__name__,
                self.module_stride,
                self.openmp
                )

    def c_compile_args(self):
        if self.openmp:
            return ['-fopenmp']
        return []


class FilterActs(Base):
    """
    Images of shape: colors x
    Filters are of shape:
        channels
    """
    @classmethod
    def infer_shape_without_instance(cls, ishape, fshape):
        icount, icolors, irows, icols = ishape
        fmodules, filters_per_module, fcolors, frows, fcols = fshape

        if not any_symbolic(irows, icols) and irows != icols:
            raise ValueError("non-square image argument",
                    (irows, icols))
        if not any_symbolic(frows, fcols) and frows != fcols:
            raise ValueError("non-square filter shape",
                    (frows, fcols))
        if (not any_symbolic(icolors, fcolors)
                and icolors != fcolors):
            raise ValueError("color counts don't match",
                    (icolors, fcolors))
        if (irows < frows or icols < fcols):
            raise ValueError("filters' size is too small",
                    (irows, icols))
        hrows = irows / frows
        hcols = icols / fcols
        hshape = (icount, fmodules, filters_per_module, hrows, hcols)
        return hshape

    def make_node(self, images, filters):
        images = theano.tensor.as_tensor_variable(images)
        filters = theano.tensor.as_tensor_variable(filters)
        ibcast = images.broadcastable
        fbcast = filters.broadcastable
        icount, icolors, irows, icols = ibcast
        fmodules, filters_per_module, fcolors, frows, fcols = fbcast  #fmodules will alone the diagonal of the images
        #print fmodules, fcolors, frows, fcols, filters_per_module
        hbcast = (icount, fmodules, filters_per_module, frows, fcols) #should be (False, False, False, False, False)
        htype = theano.tensor.TensorType(
                dtype=images.dtype,
                broadcastable=hbcast)
        if images.dtype != filters.dtype:
            raise TypeError('dtype mismatch', (images, filters))
        return theano.gof.Apply(self,
                [images, filters],
                [htype()])

    def perform(self, node, iargs, ostor):
        #print 'into FilterActs.perform'
        images, filters = iargs

        # icount : number of images in minibatch
        # icolors : number of color channel in the image ( 1=grayscale, 3=RGB, ...)
        # irows and icols : size of each image
        icount, icolors, irows, icols = images.shape
        fmodules, filters_per_module, fcolors, frows, fcols = filters.shape

        hshape = self.infer_shape(node, (images.shape, filters.shape))[0]
        _, _, _, hrows, hcols = hshape
        hidacts = numpy.zeros(hshape, dtype=images.dtype)
        for m in xrange(fmodules):
            for hR in xrange(hrows):
                img_r_offset = m * self.module_stride + hR * frows
                for hC in xrange(hcols):
                    img_c_offset = m * self.module_stride + hC * fcols
                    rc_images = images[:, :,
                            img_r_offset:img_r_offset + frows,
                            img_c_offset:img_c_offset + fcols]
                    rc_filters = filters[m]
                    # rc_images are count x fcolors x frows x fcols
                    # rc_filters are fpm x fcolors x frows x fcols
                    rc_hidacts = numpy.dot(
                        rc_images.reshape(icount, -1),
                        rc_filters.reshape(filters_per_module, -1).T
                        )
                    hidacts[:, m, :, hR, hC] = rc_hidacts
        ostor[0][0] = hidacts
        #print 'exiting FilterActs.perform'
        if 0:
            print 'FilterActs shapes: images', images.shape
            print 'FilterActs shapes: filters', filters.shape
            print 'FilterActs shapes: hidacts', hidacts.shape

    def grad(self, inputs, goutputs):
        images, filters = inputs
        _, _, _, frows, fcols = filters.shape
        _, _, irows, icols = images.shape
        gimages = ImgActs(module_stride=self.module_stride)(
                filters, goutputs[0], irows, icols)
        gfilters = WeightActs(module_stride=self.module_stride)(
                images, goutputs[0], frows, fcols)
        return [gimages, gfilters]
        
    def infer_shape(self, node, shapes):
        ishape, fshape = shapes

        icount, icolors, irows, icols = ishape
        fmodules, filters_per_module, fcolors, frows, fcols = fshape
        if not any_symbolic(irows, icols) and irows != icols:
            raise ValueError("non-square image argument",
                    (irows, icols))
        if not any_symbolic(frows, fcols) and frows != fcols:
            raise ValueError("non-square filter shape",
                    (frows, fcols))
        if (not any_symbolic(icolors, fcolors)
                and icolors != fcolors):
            raise ValueError("color counts don't match",
                    (icolors, fcolors))
        """
        if (irows < frows or icols < fcols):
            raise ValueError("filters' size is too small",
                    (irows, icols))           
        """
        hrows = irows / frows
        hcols = icols / fcols
        hshape = (icount, fmodules, filters_per_module, hrows, hcols)
        return [hshape]


class WeightActs(Base):
    """
    Images of shape: colors x
    Filters are of shape:
        channels
    """

    def make_node(self, images, hidacts, frows, fcols):
        images, hidacts, frows, fcols = map(theano.tensor.as_tensor_variable,
                [images, hidacts, frows, fcols])
        if frows.dtype[:3] not in ('int', 'uin'): #dtype is a string. should be 'int8' 'int16' 'uint8' ...
            raise TypeError(frows)
        if fcols.dtype[:3] not in ('int', 'uin'):
            raise TypeError(frows)
        if frows.ndim:
            raise TypeError('frows should be scalar', frows)
        if fcols.ndim:
            raise TypeError('fcols should be scalar', fcols)

        if images.dtype != hidacts.dtype: #should be floatX
            raise TypeError('images and hidacts dtype mismatch',
                    (images.dtype, hidacts.dtype))

        icount, icolors, irows, icols = images.type.broadcastable #should be (False, False, False, False)
        #print icolors, irows, icols, icount
        hcount, fmodules, filters_per_module, hrows, hcols = hidacts.type.broadcastable
        otype = theano.tensor.TensorType(
                dtype=images.dtype,
                broadcastable=(fmodules, filters_per_module, icolors,
                    False, False)) #frows and fcols should not be broadcastable
        return theano.Apply(self,
                [images, hidacts, frows, fcols],
                [otype()])

    def perform(self, node, iargs, ostor):
        #print 'into WeightActs.perform'
        images, hidacts, frows, fcols = iargs

        if frows != fcols:
            # this could be implemented, but GPU case doesn't do it
            raise NotImplementedError("non-square filter shape",
                    (frows, fcols))

        icount, fmodules, filters_per_module, hrows, hcols = hidacts.shape
        fshape = list(self.infer_shape(node,
                (images.shape, hidacts.shape, (), ()))[0]) #why put (frows,) and (fcols,) here
        fcolors = fshape[2]
        fshape[3] = frows
        fshape[4] = fcols

        filters = numpy.zeros(fshape, dtype=images.dtype)

        for m in xrange(fmodules):
            for hR in xrange(hrows):
                for hC in xrange(hcols):
                    img_r_offset = m*self.module_stride + hR*frows
                    img_c_offset = m*self.module_stride + hC*fcols
                    rc_images = images[:,:,
                            img_r_offset:img_r_offset + frows,
                            img_c_offset:img_c_offset + fcols]
                    # rc_images is icount x icolors x irows x icols 

                    rc_hidacts = hidacts[:, m, :, hR, hC]
                    # rc_hidacts is count x fpm 

                    rc_filters = numpy.dot(
                            rc_hidacts.T,
                            rc_images.reshape(icount, -1))
                    filters[m, :, :, :, :] += rc_filters.reshape(
                            (filters_per_module, fcolors, frows, fcols))
        ostor[0][0] = filters
        
    def c_support_code(self):
        return blas.blas_header_text()

    def c_libraries(self):
        return blas.ldflags()

    def c_compile_args(self):
        return blas.ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return blas.ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return blas.ldflags(libs=False, include_dir=True)
    
    def c_code_disabled(self, node, node_name, input_names, output_names, sub):
        
        # Extract input values
        images, hidacts, frows, fcols = input_names
        #filters, hidacts, irows, icols = input_names
        
        # Determine which BLAS function to use
        conv_type = scalar.upcast(node.inputs[0].type.dtype, 
                                  node.inputs[1].type.dtype) 
        if conv_type == 'float32':
            conv_type = "float"
            gemv = "sgemv_"
        elif conv_type == 'float64':
            conv_type = "double"
            gemv = "dgemv_"
        else:
            raise Exception()
        
        # Extract output values
        output = output_names[0]
        
        # Assign self.module_stride to a local variable else 
        # the %(module_stride)s fails
        module_stride = self.module_stride
        
        #Generate C code
        fail = sub['fail']
        sio = StringIO.StringIO()

        print >> sio, """
        
        // Validate the shape and the data type of the input tensors
        
        if (%(hidacts)s->nd != 5){
            PyErr_SetString(PyExc_ValueError, "hidacts not a 5d tensor");
            %(fail)s;
        }
        
        if (%(images)s->nd != 4){
            PyErr_SetString(PyExc_ValueError, "images not a 4d tensor");
            %(fail)s;
        }
        
        if ((%(hidacts)s->descr->type_num != PyArray_DOUBLE) && 
            (%(hidacts)s->descr->type_num != PyArray_FLOAT)){
            PyErr_SetString(PyExc_TypeError, 
                            "hidacts type should be float32 or float64");
            %(fail)s;
        }
        
        if ((%(images)s->descr->type_num != PyArray_DOUBLE) && 
            (%(images)s->descr->type_num != PyArray_FLOAT)){
            PyErr_SetString(PyExc_TypeError, 
                            "images type should be float32 or float64");
            %(fail)s;
        }
        
        if (%(images)s->descr->type_num != %(hidacts)s->descr->type_num){
            PyErr_SetString(PyExc_TypeError,
                            "images and hidacts should have the same type");
            %(fail)s;
        }
        
        {   // New scope level to avoid cross-initialization
        
            // Extract input variables
            
            int hcount = %(hidacts)s->dimensions[0];
            int fmodules = %(hidacts)s->dimensions[1];
            int filters_per_module = %(hidacts)s->dimensions[2];
            int hrows = %(hidacts)s->dimensions[3];
            int hcols = %(hidacts)s->dimensions[4];
                       
            int icount = %(images)s->dimensions[0];
            int icolors = %(images)s->dimensions[1];
            int irows = %(images)s->dimensions[2];
            int icols = %(images)s->dimensions[3];
            
            int frows = ((dtype_%(frows)s *) (%(frows)s->data))[0];
            int fcols = ((dtype_%(fcols)s *) (%(fcols)s->data))[0];
            
            int module_stride = %(module_stride)s;
            
            
            // Validate the shape of the input tensors
            
            if ( hrows != hcols ){
                PyErr_SetString(PyExc_ValueError, 
                                "non-square hidacts argument");
                %(fail)s;
            }
            
            if ( frows != fcols ){
                PyErr_SetString(PyExc_ValueError, 
                                "non-square filter shape");
                %(fail)s;
            }
            
            if ( irows != icols ){
                PyErr_SetString(PyExc_ValueError, 
                                "non-square image argument");
                %(fail)s;
            }
            
            if ( hcount != icount ){
                PyErr_SetString(PyExc_ValueError,
                                "inconsistent batch size");
                %(fail)s;
            }
                       
            if (hrows * frows + fmodules - 1 != irows){
                PyErr_SetString(
                      PyExc_ValueError,
                      "hrows * frows + fmodules - 1 should be equal to irows");
                %(fail)s;
            }
            
            if (hcols * fcols + fmodules - 1 != icols){
                PyErr_SetString(
                    PyExc_ValueError,
                    "hcols * fcols + fmodules - 1 should be equal to icols");
                %(fail)s;
            }
            
                                
            // Ensure output array is of the proper format
            
            if (NULL == %(output)s || 
               (%(output)s->dimensions[0] != fmodules) || 
               (%(output)s->dimensions[1] != filters_per_module) || 
               (%(output)s->dimensions[2] != icolors) || 
               (%(output)s->dimensions[3] != frows) || 
               (%(output)s->dimensions[4] != fcols) ||
               (!PyArray_ISBEHAVED(%(output)s)) || 
               ((%(output)s->descr->type_num != PyArray_DOUBLE) && 
                (%(output)s->descr->type_num != PyArray_FLOAT)))
            {
                // The output array is of an invalid format.
                
                if (NULL != %(output)s) Py_XDECREF(%(output)s);
                
                npy_intp outputDims[5];
                outputDims[0] = fmodules;
                outputDims[1] = filters_per_module;
                outputDims[2] = icolors;
                outputDims[3] = frows;
                outputDims[4] = fcols;
            
                %(output)s = (PyArrayObject*)PyArray_ZEROS(5, outputDims, 
                                             %(images)s->descr->type_num, 0);
                if(!%(output)s) {
                    PyErr_SetString(PyExc_MemoryError, 
                                    "failed to alloc memory for output");
                    %(fail)s
                }
                
   
            }else{
            
                // The output array is of the proper format. 
                // Its content must be initialized to zeros.
                
                dtype_%(output)s* data_ptr = 
                                (dtype_%(output)s*)PyArray_DATA(%(output)s);
                                
                npy_intp s0 = PyArray_STRIDE(%(output)s, 0) /
                                            PyArray_ITEMSIZE(%(output)s);
                npy_intp s1 = PyArray_STRIDE(%(output)s, 1) /
                                             PyArray_ITEMSIZE(%(output)s);
                npy_intp s2 = PyArray_STRIDE(%(output)s, 2) /
                                             PyArray_ITEMSIZE(%(output)s);
                npy_intp s3 = PyArray_STRIDE(%(output)s, 3) / 
                                            PyArray_ITEMSIZE(%(output)s);
                npy_intp s4 = PyArray_STRIDE(%(output)s, 4) /
                                            PyArray_ITEMSIZE(%(output)s);
                                
                for(int module = 0; module < fmodules; module++){
                    for(int fpm = 0; fpm < fmodules; fpm++){
                        for(int color = 0; color < icolors; color++){
                            for(int row = 0; row < irows; row++){
                                for(int col = 0; col < icols; col++){
                            
                                    data_ptr[module * s0 + fpm * s1 +
                                             color * s2 + row * s3 + 
                                             row * s4] = 0.0f;
                                }
                            }
                        }
                    }
                }  
            }       
            
            
            // Extract the arrays' strides
            
            npy_intp hidacts_count_stride = PyArray_STRIDE(%(hidacts)s, 0) /
                                            PyArray_ITEMSIZE(%(hidacts)s);
            npy_intp hidacts_module_stride = PyArray_STRIDE(%(hidacts)s, 1) /
                                               PyArray_ITEMSIZE(%(hidacts)s);
            npy_intp hidacts_filter_stride = PyArray_STRIDE(%(hidacts)s, 2) /
                                             PyArray_ITEMSIZE(%(hidacts)s);
            npy_intp hidacts_hrows_stride = PyArray_STRIDE(%(hidacts)s, 3) / 
                                            PyArray_ITEMSIZE(%(hidacts)s);
            npy_intp hidacts_hcols_stride = PyArray_STRIDE(%(hidacts)s, 4) /
                                            PyArray_ITEMSIZE(%(hidacts)s);
                      
            npy_intp images_count_stride = PyArray_STRIDE(%(images)s, 0) /
                                           PyArray_ITEMSIZE(%(images)s);
            npy_intp images_color_stride = PyArray_STRIDE(%(images)s, 1) /
                                           PyArray_ITEMSIZE(%(images)s);
            npy_intp images_irows_stride = PyArray_STRIDE(%(images)s, 2) /
                                           PyArray_ITEMSIZE(%(images)s);
            npy_intp images_icols_stride = PyArray_STRIDE(%(images)s, 3) /
                                           PyArray_ITEMSIZE(%(images)s);
                                           
            npy_intp output_module_stride = PyArray_STRIDE(%(output)s, 0) /
                                              PyArray_ITEMSIZE(%(output)s);
            npy_intp output_filter_stride = PyArray_STRIDE(%(output)s, 1) /
                                             PyArray_ITEMSIZE(%(output)s);
            npy_intp output_color_stride = PyArray_STRIDE(%(output)s, 2) /
                                             PyArray_ITEMSIZE(%(output)s);
            npy_intp output_frows_stride = PyArray_STRIDE(%(output)s, 3) /
                                            PyArray_ITEMSIZE(%(output)s);
            npy_intp output_fcols_stride = PyArray_STRIDE(%(output)s, 4) /
                                            PyArray_ITEMSIZE(%(output)s);
                          
                          
            // Check if BLAS' gemv can be used to speed up the computations
            
            bool useBlas = PyArray_ISCONTIGUOUS(%(hidacts)s) &&
                           PyArray_ISCONTIGUOUS(%(images)s);
                     
            
            
            // Allocate memory for the array in which the content of images
            // will be copied so that it will be C Contiguous for BLAS' gemv
            
            npy_intp dotPDims[2];
            dotPDims[0] = icount;
            dotPDims[1] = icolors * frows * fcols;
            
            PyArrayObject* img_C = 
                    (PyArrayObject*)PyArray_ZEROS(2, dotPDims,
                                                  %(output)s->descr->type_num,
                                                  0);
            if(!img_C) {
                PyErr_SetString(PyExc_MemoryError, 
                                "failed to alloc memory for dotPResult");
                %(fail)s
            }
            dtype_%(output)s* img_C_ptr = (dtype_%(output)s*)(img_C->data);
                
          
        
            // Allocate variable used to call the BLAS function
            
            char noTrans = 'N';
            %(conv_type)s alpha = 1.0f;
            %(conv_type)s beta = 1.0f;
            int nbRowsImages = icount;
            int nbColsImages = icolors * frows * fcols;
            int LDA = icolors * frows * fcols;
            int hidacts_inc = hidacts_count_stride;
            int inc_output = 1; // because %(output)s is C-contiguous
                
                
            // Compute the output     
            
            dtype_%(hidacts)s* hidacts_ptr = 
                                (dtype_%(hidacts)s*)PyArray_DATA(%(hidacts)s);
            dtype_%(images)s* images_ptr = 
                                (dtype_%(images)s*)PyArray_DATA(%(images)s);
            dtype_%(output)s* output_ptr = 
                                (dtype_%(output)s*)PyArray_DATA(%(output)s);
        
            
            for(int m=0; m < fmodules; m++){
            
                hidacts_ptr += m * hidacts_module_stride;
                output_ptr += m * output_module_stride;
             
            
                for(int hR=0; hR < hrows; hR++){
                
                    hidacts_ptr += hR * hidacts_hrows_stride;
                    int img_r_offset = m * module_stride + hR * frows;
                    
                
                    for(int hC=0; hC < hcols; hC++){
                    
                        hidacts_ptr += hC * hidacts_hcols_stride;
                        int img_c_offset = m * module_stride + hC * frows;
                        
                        if(useBlas){
                        
                            // Use BLAS' gemv function to speed up 
                            // the calculation of the dot products.
                            
                            /****************************/
                            /****       TODO       ******/
                            /****************************/
                            
                            
                            for(int fpm=0; fpm < filters_per_module; fpm++){                           
                            
                                hidacts_ptr += fpm * hidacts_filter_stride;
                                output_ptr += fpm * output_filter_stride;
                                
                                // Copy the relevant data from images into
                                // the img_C array
                                for(int icountIndex=0; icountIndex < icount; 
                                    icountIndex++){
                                    
                                    images_ptr += icountIndex * 
                                                  images_count_stride;
                                
                                    for(int icolorsIndex=0; icolorsIndex < 
                                        icolors; icolorsIndex++){
                                        
                                        images_ptr += icolorsIndex * 
                                                    images_color_stride;
                                        
                                        for(int frowsIndex=0; 
                                            frowsIndex < frows; 
                                            frowsIndex++){
                                        
                                            images_ptr += 
                                                    (img_r_offset + 
                                                     frowsIndex) * 
                                                    images_irows_stride;
                                        
                                            for(int fcolsIndex=0; 
                                                fcolsIndex < fcols; 
                                                fcolsIndex++){
                                            
                                                images_ptr += 
                                                    (img_c_offset + 
                                                     fcolsIndex) * 
                                                    images_icols_stride;
                                                                                            
                                                img_C_ptr[icountIndex * 
                                                          icolors * 
                                                          frows * fcols +
                                                          icolorsIndex * 
                                                          frows * fcols + 
                                                          frowsIndex * fcols +
                                                          fcolsIndex] = images_ptr[0];
                                                                                               
                                                images_ptr -= 
                                                    (img_c_offset + 
                                                     fcolsIndex) * 
                                                    images_icols_stride;
                                            }
                                            
                                            images_ptr -= 
                                                    (img_r_offset + 
                                                     frowsIndex) * 
                                                    images_irows_stride;
                                        }
                                        
                                        images_ptr -= icolorsIndex * 
                                                    images_color_stride;
                                    }
                                    
                                    images_ptr -= icountIndex * 
                                                  images_count_stride;
                                }
                                
                                
                                
                                //std::cout<<output_ptr[0] << std::endl;
                                //raise(SIGINT);
                                
                                // Perform the dot product                                
                                %(gemv)s(&noTrans, &nbColsImages,
                                         &nbRowsImages, &alpha,
                                         img_C_ptr, &LDA,
                                         hidacts_ptr, &hidacts_inc,
                                         &beta, output_ptr, &inc_output);
                                         
                                hidacts_ptr -= fpm * hidacts_filter_stride;
                                output_ptr -= fpm * output_filter_stride;
                            
                            }
                            
                        }else{
                        
                            // Use a slow BLAS-free version
                            
                            /****************************/
                            /****       TODO       ******/
                            /****************************/ 
                        
                        }
                        
                        hidacts_ptr -= hC * hidacts_hcols_stride;
                    }
                    
                    hidacts_ptr -= hR * hidacts_hrows_stride;
                }
                
                hidacts_ptr -= m * hidacts_module_stride;
                output_ptr -= m * output_module_stride;
            }
        
            /*
                        rc_images = images[:,:,
                                img_r_offset:img_r_offset + frows,
                                img_c_offset:img_c_offset + fcols]
                        # rc_images is icount x icolors x irows x icols 

                        rc_hidacts = hidacts[:, m, :, hR, hC]
                        # rc_hidacts is count x fpm 

                        rc_filters = numpy.dot(
                                rc_hidacts.T,
                                rc_images.reshape(icount, -1))
                        filters[m, :, :, :, :] += rc_filters.reshape(
                                (filters_per_module, fcolors, frows, fcols))
        
            */
        
        }
        
        """
               
        return sio.getvalue() % locals()    

    def grad(self, inputs, goutputs):
        images, hidacts, frows, fcols = inputs
        gfilters, = goutputs
        _, _, irows, icols = images.shape
        gimages = ImgActs(module_stride=self.module_stride)(
                gfilters, hidacts, irows, icols)
        ghidacts = FilterActs(module_stride=self.module_stride)(
                images, gfilters)
        return [gimages, ghidacts, None, None]       

    def infer_shape(self, node, shapes):
        images, hidacts, frows, fcols = node.inputs
        ishape, hshape, frowshp, fcolshp = shapes
        icount, icolors, irows, icols = ishape
        hcount, fmodules, filters_per_module, hrows, hcols = hshape
        
        fcolors = icolors
        # frows already assigned
        # fcols already assigned
        fshape = (fmodules, filters_per_module, fcolors, frows, fcols )

        if not_symbolic(irows, icols) and irows != icols:
            raise NotImplementedError("non-square image argument",
                    (irows, icols))
        if not_symbolic(hrows, hcols) and hrows != hcols:
            raise NotImplementedError("non-square filter shape",
                    (hrows, hcols))
        if not_symbolic(icount, hcount) and icount != hcount:
            raise NotImplementedError("different number of images",
                    (icount, hcount))
       
        return [fshape]


class ImgActs(Base):
    """
    XXX
    """
    def make_node(self, filters, hidacts, irows, icols):
        filters, hidacts, irows, icols = map(theano.tensor.as_tensor_variable,
                [filters, hidacts, irows, icols])
        if irows.dtype[:3] not in ('int', 'uin'):
            raise TypeError(irows)
        if icols.dtype[:3] not in ('int', 'uin'):
            raise TypeError(irows)
        if irows.ndim:
            raise TypeError('irows should be scalar', irows)
        if icols.ndim:
            raise TypeError('icols should be scalar', icols)
        if filters.ndim != 5: #(fmodules, filters_per_module, fcolors, frows, fcols)
            raise TypeError('filters must be 7d tensor', filters)
        if hidacts.ndim != 5: #(icount, fmodules, filters_per_module, hrows, hcols)
            raise TypeError('hidacts must be 5d tensor', filters)
        if filters.dtype != hidacts.dtype: #should be floatX
            raise TypeError('filters and hidacts must have matching dtype',
                    (filters, hidacts))
        hcount, fmodules, filters_per_module, hrows, hcols = hidacts.type.broadcastable
        #print fmodules, filters_per_module, hrows, hcols, hcount
        #print hidacts
        _, _, fcolors, _, _ = filters.type.broadcastable
        
        otype = theano.tensor.TensorType(
                dtype=filters.dtype,
                broadcastable=(hcount, fcolors,
                    False, False)) # irows and icols should not be broadcastable
        return theano.gof.Apply(self,
                [filters, hidacts, irows, icols],
                [otype()])

    def perform(self, node, iargs, ostor):
        #print 'into ImgActs.perform' 
        filters, hidacts, irows, icols = iargs

        # hcount : minibatch size (nb image passed)
        # fmodules : For one position, how many filters
        hcount, fmodules, filters_per_module, hrows, hcols = hidacts.shape

        # fmodules : nb of modules ( module = group of non-overlaping filters )
        # filters per module : nomber of filters on each position ('looking' at the same image area)
        # fcolors : nb of color channels ( 1 for grayscale, 3 for RGB, ... )
        # frows x fcols : size of filter
        fmodules_, filters_per_module_, fcolors, frows, fcols = filters.shape
        
        assert fmodules_==fmodules
        assert filters_per_module_==filters_per_module

        icolors = fcolors
        icount = hcount

        #print 'IMGACTS: NODE OUTPUTS[0]'
        #print theano.printing.debugprint(node.outputs[0])
        #print 'FILTERS SHAPE:', filters.shape
        #print 'HIDACTS SHAPE:', hidacts.shape
        if hrows != hcols:
            raise NotImplementedError("non-square hidacts argument",
                    (hrows, hcols))
        if frows != fcols:
            raise NotImplementedError("non-square filter shape",
                    (frows, fcols))
        if irows != icols:
            raise NotImplementedError("non-square image argument",
                    (irows, icols))
        if hrows * frows + fmodules - 1 != irows:
            raise NotImplementedError("hrows * frows + fmodules - 1 should" +
                                      "be equal to irows",
                                      (hrows * frows + fmodules - 1, irows))
        if hcols * fcols + fmodules - 1 != icols:
            raise NotImplementedError("hcols * fcols + fmodules - 1 should" +
                                      "be equal to icols",
                                      (hcols * fcols + fmodules - 1, icols))

        images = numpy.zeros(
                (icount, icolors, irows, icols),
                dtype=hidacts.dtype)
                
        for m in xrange(fmodules):
            for hR in xrange(hrows):
                img_r_offset = m*self.module_stride + hR*frows
                for hC in xrange(hcols):
                    rc_filters = filters[m, :, :, :, :]
                    # rc_filters is fpm x fcolors x frows x fcols

                    rc_hidacts = hidacts[:, m, :, hR, hC]
                    # rc_hidacts is icount x fpm 
                    
                    img_c_offset = m*self.module_stride + hC*fcols
                                        
                    images[:,:,
                            img_r_offset:img_r_offset + frows,
                            img_c_offset:img_c_offset + fcols
                            ] += numpy.dot(
                                    rc_hidacts,
                                    rc_filters.reshape(filters_per_module, -1)                                    
                                    ).reshape(
                                    (icount, fcolors, frows, fcols))
        ostor[0][0] = images
        #print 'exiting ImgActs perform'
    
    def c_support_code(self):
        return blas.blas_header_text()

    def c_libraries(self):
        return blas.ldflags()

    def c_compile_args(self):
        ret = blas.ldflags(libs=False, flags=True)
        if self.openmp:
            ret += ['-fopenmp']
        return ret

    def c_lib_dirs(self):
        return blas.ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return blas.ldflags(libs=False, include_dir=True)
    
    def c_code(self, node, node_name, input_names, output_names, sub):
        
        # Extract input values
        filters, hidacts, irows, icols = input_names
        
        # Determine which BLAS function to use
        conv_type = scalar.upcast(node.inputs[0].type.dtype, 
                                  node.inputs[1].type.dtype) 
        if conv_type == 'float32':
            conv_type = "float"
            gemv = "sgemv_"
        elif conv_type == 'float64':
            conv_type = "double"
            gemv = "dgemv_"
        else:
            raise Exception()
        
        # Extract output values
        output = output_names[0]
        
        # Assign self.module_stride to a local variable else 
        # the %(module_stride)s fails
        module_stride = self.module_stride
        
        #Generate C code
        fail = sub['fail']
        sio = StringIO.StringIO()

        print >> sio, """
        
        // Validate the shape and the data type of the input tensors
        
        if (%(hidacts)s->nd != 5){
            PyErr_SetString(PyExc_ValueError, "hidacts not a 5d tensor");
            %(fail)s;
        }
        
        if (%(filters)s->nd != 5){
            PyErr_SetString(PyExc_ValueError, "filters not a 5d tensor");
            %(fail)s;
        }
        
        if ((%(hidacts)s->descr->type_num != PyArray_DOUBLE) && 
            (%(hidacts)s->descr->type_num != PyArray_FLOAT)){
            PyErr_SetString(PyExc_TypeError, 
                            "hidacts type should be float32 or float64");
            %(fail)s;
        }
        
        if ((%(filters)s->descr->type_num != PyArray_DOUBLE) && 
            (%(filters)s->descr->type_num != PyArray_FLOAT)){
            PyErr_SetString(PyExc_TypeError, 
                            "filters type should be float32 or float64");
            %(fail)s;
        }
        
        if (%(filters)s->descr->type_num != %(hidacts)s->descr->type_num){
            PyErr_SetString(PyExc_TypeError,
                            "filters and hidacts should have the same type");
            %(fail)s;
        }
        
        {   // New scope level to avoid cross-initialization
        
            // Extract input variables
            
            int hcount = %(hidacts)s->dimensions[0];
            int fmodules = %(hidacts)s->dimensions[1];
            int filters_per_module = %(hidacts)s->dimensions[2];
            int hrows = %(hidacts)s->dimensions[3];
            int hcols = %(hidacts)s->dimensions[4];
            
            int fmodules_ = %(filters)s->dimensions[0];
            int filters_per_module_ = %(filters)s->dimensions[1];
            int fcolors = %(filters)s->dimensions[2];
            int frows = %(filters)s->dimensions[3];
            int fcols = %(filters)s->dimensions[4];
            
            int irows = ((dtype_%(irows)s *) (%(irows)s->data))[0];
            int icols = ((dtype_%(icols)s *) (%(icols)s->data))[0];
            
            int module_stride = %(module_stride)s;
            
            
            // Validate the shape of the input tensors
            
            if (hrows != hcols){
                PyErr_SetString(PyExc_ValueError, 
                                "non-square hidacts argument");
                %(fail)s;
            }
            
            if (frows != fcols){
                PyErr_SetString(PyExc_ValueError, 
                                "non-square filter shape");
                %(fail)s;
            }
            
            if (irows != icols){
                PyErr_SetString(PyExc_ValueError, 
                                "non-square image argument");
                %(fail)s;
            }
            
            if (fmodules_ != fmodules){
                PyErr_SetString(PyExc_ValueError,
                                "inconsistent number of filter modules");
                %(fail)s;
            }
            
            if (filters_per_module_ != filters_per_module){
                PyErr_SetString(PyExc_ValueError,
                                "inconsistent number of filters by modules");
                %(fail)s;
            }
            
            if (hrows * frows + fmodules - 1 != irows){
                PyErr_SetString(
                      PyExc_ValueError,
                      "hrows * frows + fmodules - 1 should be equal to irows");
                %(fail)s;
            }
            
            if (hcols * fcols + fmodules - 1 != icols){
                PyErr_SetString(
                    PyExc_ValueError,
                    "hcols * fcols + fmodules - 1 should be equal to icols");
                %(fail)s;
            }
            
            
                    
            // Ensure output array is of the proper format
            
            if (NULL == %(output)s || 
                    (%(output)s->dimensions[0] != hcount) || 
                    (%(output)s->dimensions[1] != fcolors) || 
                    (%(output)s->dimensions[2] != irows) || 
                    (%(output)s->dimensions[3] != icols) || 
                    (!PyArray_ISBEHAVED(%(output)s)) || 
                    ((%(output)s->descr->type_num != PyArray_DOUBLE) && 
                     (%(output)s->descr->type_num != PyArray_FLOAT)))
            {
                // The output array is of an invalid format.
                
                if (NULL != %(output)s) Py_XDECREF(%(output)s);
                
                npy_intp outputDims[4];
                outputDims[0] = hcount;
                outputDims[1] = fcolors;
                outputDims[2] = irows;
                outputDims[3] = icols;
            
                %(output)s = (PyArrayObject*)PyArray_ZEROS(4, outputDims, 
                                             %(filters)s->descr->type_num, 0);
                if(!%(output)s) {
                    PyErr_SetString(PyExc_MemoryError, 
                                    "failed to alloc memory for output");
                    %(fail)s
                }
                
            }else{
            
                // The output array is of the proper format. 
                // Its content must be initialized to zeros.
                
                for(int count=0; count < hcount; count++){
                    for(int color=0; color < fcolors; color++){
                        for(int row=0; row < irows; row++){
                            for(int col=0; col < icols; col++){
                                ((dtype_%(output)s*) 
                                        PyArray_GETPTR4(%(output)s, count,
                                                        color, row, 
                                                        col))[0] = 0.0f;     
                            }
                        }
                    }
                }
                
            }
            
            
            // Extract the arrays' strides
            
            npy_intp hidacts_count_stride = PyArray_STRIDE(%(hidacts)s, 0) /
                                            PyArray_ITEMSIZE(%(hidacts)s);
            npy_intp hidacts_fmodule_stride = PyArray_STRIDE(%(hidacts)s, 1) /
                                               PyArray_ITEMSIZE(%(hidacts)s);
            npy_intp hidacts_filter_stride = PyArray_STRIDE(%(hidacts)s, 2) /
                                             PyArray_ITEMSIZE(%(hidacts)s);
            npy_intp hidacts_hrows_stride = PyArray_STRIDE(%(hidacts)s, 3) / 
                                            PyArray_ITEMSIZE(%(hidacts)s);
            npy_intp hidacts_hcols_stride = PyArray_STRIDE(%(hidacts)s, 4) /
                                            PyArray_ITEMSIZE(%(hidacts)s);
            
            npy_intp filters_fmodule_stride = PyArray_STRIDE(%(filters)s, 0) /
                                              PyArray_ITEMSIZE(%(filters)s);
            npy_intp filters_filter_stride = PyArray_STRIDE(%(filters)s, 1) /
                                             PyArray_ITEMSIZE(%(filters)s);
            npy_intp filters_fcolor_stride = PyArray_STRIDE(%(filters)s, 2) /
                                             PyArray_ITEMSIZE(%(filters)s);
            npy_intp filters_frows_stride = PyArray_STRIDE(%(filters)s, 3) /
                                            PyArray_ITEMSIZE(%(filters)s);
            npy_intp filters_fcols_stride = PyArray_STRIDE(%(filters)s, 4) /
                                            PyArray_ITEMSIZE(%(filters)s);
            
            npy_intp output_count_stride = PyArray_STRIDE(%(output)s, 0) /
                                           PyArray_ITEMSIZE(%(output)s);
            npy_intp output_color_stride = PyArray_STRIDE(%(output)s, 1) /
                                           PyArray_ITEMSIZE(%(output)s);
            npy_intp output_frows_stride = PyArray_STRIDE(%(output)s, 2) /
                                           PyArray_ITEMSIZE(%(output)s);
            npy_intp output_fcols_stride = PyArray_STRIDE(%(output)s, 3) /
                                           PyArray_ITEMSIZE(%(output)s);
            
            
            // Check if BLAS' gemv can be used to speed up the computations
            
            bool useBlas = PyArray_ISCONTIGUOUS(%(hidacts)s) &&
                           PyArray_ISCONTIGUOUS(%(filters)s);
                     
            
            // Allocate memory for the result of the dot product
            
            npy_intp dotPDims[3];
            dotPDims[0] = fcolors * frows * fcols;
            
            PyArrayObject* dotPResult = 
                    (PyArrayObject*)PyArray_ZEROS(1, dotPDims,
                                                  %(output)s->descr->type_num,
                                                  0);
            if(!dotPResult) {
                PyErr_SetString(PyExc_MemoryError, 
                                "failed to alloc memory for dotPResult");
                %(fail)s
            }
            dtype_%(output)s* dotp = (dtype_%(output)s*)(dotPResult->data);
            
            
            // Allocate variable used to call the BLAS function
            
            char noTrans = 'N';
            %(conv_type)s alpha = 1.0f;
            %(conv_type)s beta = 0.0f;
            int nbRowsFilters = filters_per_module;
            int nbColsFilters = fcolors * frows * fcols;
            int LDA = fcolors * frows * fcols;
            int hidacts_inc = hidacts_filter_stride;
            int inc_output = 1; // because dotPResult is C-contiguous
                
                
            // Compute the output     
            
            dtype_%(hidacts)s* hidacts_ptr = 
                                (dtype_%(hidacts)s*)PyArray_DATA(%(hidacts)s);
            dtype_%(filters)s* filters_ptr = 
                                (dtype_%(filters)s*)PyArray_DATA(%(filters)s);
            dtype_%(hidacts)s* output_ptr = 
                                (dtype_%(output)s*)PyArray_DATA(%(output)s);

//We swap the loop on hrows and fmodules as we can't parallelize on
//fmodules as this create multiple write to the same adress by
//multiple threads.
#pragma omp parallel for schedule(static) firstprivate(hidacts_ptr, filters_ptr, output_ptr)
            for(int hR=0; hR < hrows; hR++){
                hidacts_ptr += hR * hidacts_hrows_stride;

                for(int m=0; m < fmodules; m++){
                    hidacts_ptr += m * hidacts_fmodule_stride;
                    filters_ptr += m * filters_fmodule_stride;
                    int img_r_offset = m * module_stride + hR * frows;
                    
                
                    for(int hC=0; hC < hcols; hC++){
                    
                        hidacts_ptr += hC * hidacts_hcols_stride;
                        int img_c_offset = m * module_stride + hC * frows;
                        
                        if(useBlas){
                        
                            // Use BLAS' gemv function to speed up 
                            // the calculation of the dot products.
                        
                            for(int icountIndex=0; icountIndex < hcount; 
                                icountIndex++){
                                                            
                                hidacts_ptr += icountIndex * 
                                               hidacts_count_stride;
                                output_ptr += icountIndex * 
                                              output_count_stride;
                                
                                %(gemv)s(&noTrans, &nbColsFilters,
                                         &nbRowsFilters, &alpha,
                                         filters_ptr, &LDA,
                                         hidacts_ptr, &hidacts_inc,
                                         &beta, dotp, &inc_output);
                                        
                                // Copy dotp content to output array
                                for(int fcolorsIndex=0; fcolorsIndex < 
                                    fcolors; fcolorsIndex++){
                                    
                                    output_ptr += fcolorsIndex * 
                                                  output_color_stride;
                                    
                                    for(int frowsIndex=0; frowsIndex < frows; 
                                        frowsIndex++){
                                    
                                        output_ptr += 
                                                (img_r_offset + frowsIndex) * 
                                                output_frows_stride;
                                    
                                        for(int fcolsIndex=0; 
                                            fcolsIndex < fcols; fcolsIndex++){
                                        
                                            output_ptr += 
                                                (img_c_offset+fcolsIndex) * 
                                                output_fcols_stride;
                                        
                                            output_ptr[0] += 
                                                dotp[fcolorsIndex * frows * 
                                                fcols + frowsIndex * fcols +
                                                fcolsIndex];
                                            
                                            output_ptr -= 
                                                (img_c_offset+fcolsIndex) * 
                                                output_fcols_stride;
                                        }
                                        
                                        output_ptr -= 
                                                (img_r_offset + frowsIndex) * 
                                                output_frows_stride;
                                    }
                                    
                                    output_ptr -= fcolorsIndex * 
                                                  output_color_stride;
                                }
                                
                                hidacts_ptr -= icountIndex * 
                                               hidacts_count_stride;
                                output_ptr -= icountIndex * 
                                              output_count_stride;
                            }
                        
                        }else{
                            // Use a slower non-BLAS version
                        
                            for(int icountIndex=0; icountIndex < hcount;
                                icountIndex++){
                            
                                hidacts_ptr += icountIndex * 
                                               hidacts_count_stride;
                                output_ptr += icountIndex * 
                                              output_count_stride;
                            
                                
                                for(int fcolorsIndex=0; 
                                    fcolorsIndex < fcolors; fcolorsIndex++){
                                
                                    filters_ptr += fcolorsIndex *
                                                   filters_fcolor_stride;
                                    output_ptr += fcolorsIndex * 
                                                  output_color_stride;
                                    
                                
                                    for(int frowsIndex=0; frowsIndex < frows;
                                        frowsIndex++){
                                    
                                        filters_ptr += 
                                                frowsIndex * 
                                                filters_frows_stride;      
                                        output_ptr += 
                                                (img_r_offset + frowsIndex) * 
                                                output_frows_stride;
                                    
                                        for(int fcolsIndex=0; 
                                            fcolsIndex < fcols; fcolsIndex++){
                                        
                                            filters_ptr += 
                                                    fcolsIndex * 
                                                    filters_fcols_stride;
                                                    
                                            output_ptr += 
                                                    (img_c_offset +
                                                     fcolsIndex) * 
                                                    output_fcols_stride;
                                            
                                            for(int filter=0; 
                                                filter < filters_per_module_; 
                                                filter++){
                                                                                    
                                                output_ptr[0] += 
                                                    *(hidacts_ptr + filter * 
                                                      hidacts_filter_stride) *
                                                    *(filters_ptr + filter *
                                                      filters_filter_stride);
                                                                                
                                            }
                                            
                                            filters_ptr -= 
                                                    fcolsIndex * 
                                                    filters_fcols_stride;
                                            output_ptr -= 
                                                    (img_c_offset + 
                                                     fcolsIndex) * 
                                                    output_fcols_stride;
                                        }
                                        
                                        filters_ptr -= frowsIndex * 
                                                       filters_frows_stride;
                                        output_ptr -= (img_r_offset + 
                                                       frowsIndex) * 
                                                      output_frows_stride;
                                    }
                                    
                                    filters_ptr -= fcolorsIndex * 
                                                   filters_fcolor_stride;
                                    output_ptr -= fcolorsIndex * 
                                                  output_color_stride;
                                }
                                
                                hidacts_ptr -= icountIndex * 
                                               hidacts_count_stride;
                                output_ptr -= icountIndex * 
                                              output_count_stride;
                            }
                        
                        }
                        
                        hidacts_ptr -= hC * hidacts_hcols_stride;
                    }
                    hidacts_ptr -= m * hidacts_fmodule_stride;
                    filters_ptr -= m * filters_fmodule_stride;
                }
                hidacts_ptr -= hR * hidacts_hrows_stride;
            }
            
        
        }

        """
               
        return sio.getvalue() % locals()
                                    

    def grad(self, inputs, goutputs):
        filters, hidacts, irows, icols = inputs
        gimages, = goutputs
        _, _, _, frows, fcols = filters.shape
        gfilters = WeightActs(module_stride=self.module_stride)(
                gimages, hidacts, frows, fcols)
        ghidacts = FilterActs(module_stride=self.module_stride)(
                gimages, filters)
        return [gfilters, ghidacts, None, None]
