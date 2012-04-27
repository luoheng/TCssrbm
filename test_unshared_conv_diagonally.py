import theano
import numpy
from theano import tensor as T
from theano import function
from theano.tensor import nnet
from theano.tensor.nnet.conv import conv2d
from theano.tensor.nnet.conv import ConvOp
from theano import gradient

from unshared_conv_diagonally import FilterActs
from unshared_conv_diagonally import WeightActs
from unshared_conv_diagonally import ImgActs

import time

floatX=theano.config.floatX
ftensor5 = theano.tensor.TensorType(floatX, (False,)*5)
arrayX = lambda X : numpy.asarray(X, dtype=floatX)

def rand(shp, dtype=floatX):
    return numpy.random.rand(*shp).astype(dtype)
    

fmodules = 2
filters_per_module = 2
icorlors = 1
irows = 5
icols = 5
icount = 1
frows = 2
fcols = 2

ishape = (icount, icorlors, irows, icols)
fshape = (fmodules, filters_per_module, icorlors, frows, fcols)
h_out_shape = FilterActs.infer_shape_without_instance(ishape, fshape)
print h_out_shape

module_stride = 1
op = FilterActs(module_stride)

img = theano.tensor.ftensor4('img')
filters = ftensor5()

h_out = op(img,filters)
fn_h_out = theano.function([img,filters],h_out)

img_value = rand(ishape)
filters_value = rand(fshape)
#filters_value = arrayX(numpy.ones(fshape))

h_out_value = fn_h_out(img_value,filters_value)
"""
print 'img'
print img_value[0,:,:,0]
"""

w=T.ftensor4('w_name')
#img_2 = img.dimshuffle(3,0,1,2)
vw_slow = conv2d(img,w)
fn_vw_slow = theano.function([img,w],vw_slow)


w_value = arrayX(numpy.zeros((fmodules*filters_per_module,icorlors,frows, fcols)))
for ii in xrange(fmodules):
    for jj in xrange(filters_per_module):
        temp = filters_value[ii,jj,0,:,:]
        w_value[ii*filters_per_module+jj,0,:,:] = temp[::-1,::-1]

vw_slow_value=fn_vw_slow(img_value,w_value)
"""
print time.strftime('%X %x %Z')
for iii in xrange(200):
    vw_slow_value=fn_vw_slow(img_value,w_value)

print time.strftime('%X %x %Z')
for iii in xrange(200):
    h_out_value = fn_h_out(img_value,filters_value)
print time.strftime('%X %x %Z')
"""


print 'filters'
print filters_value[0,0,0,:,:]
print 'w'
print w_value[0,0,:,:]
print 'h'
print h_out_value.shape
print h_out_value[0,0,0,:,:]
print h_out_value[0,0,1,:,:]
print h_out_value[0,1,0,:,:]
print h_out_value[0,1,1,:,:]

print 'slow'
print vw_slow_value.shape
print vw_slow_value[0,0]
print vw_slow_value[0,1]
print vw_slow_value[0,2]
print vw_slow_value[0,3]



mask=-1e8*arrayX(numpy.ones((icount,fmodules*filters_per_module,irows-frows+1,icols-fcols+1)))

for ii in range(fmodules):
        for jj in range(filters_per_module):
            mask[:,ii*filters_per_module+jj, \
                ii::frows, \
                ii::fcols] = 0

vw_slow_sigmoid = nnet.sigmoid(T.add(vw_slow,mask))
fn_vw_slow_sigmoid = function([img,w],vw_slow_sigmoid)
vw_slow_sigmoid_value = fn_vw_slow_sigmoid(img_value,w_value)

vw_fast_sigmoid = nnet.sigmoid(h_out)
fn_vw_fast_sigmoid = function([img,filters],vw_fast_sigmoid)
vw_fast_sigmoid_value = fn_vw_fast_sigmoid(img_value,filters_value)


print 'fast_sigmoid'
print vw_slow_sigmoid_value.shape
print vw_fast_sigmoid_value[0,0,0,:,:]
print vw_fast_sigmoid_value[0,0,1,:,:]
print vw_fast_sigmoid_value[0,1,0,:,:]
print vw_fast_sigmoid_value[0,1,1,:,:]
print 'slow_sigmoid'
print vw_fast_sigmoid_value.shape
print vw_slow_sigmoid_value[0,0]
print vw_slow_sigmoid_value[0,1]
print vw_slow_sigmoid_value[0,2]
print vw_slow_sigmoid_value[0,3]


costvw_slow = nnet.softplus(vw_slow+mask)
gvw_slow = T.grad(costvw_slow.sum(),w,consider_constant=[img])
fn_gvw_slow = function([img,w],gvw_slow)
gvw_slow_value = fn_gvw_slow(img_value,w_value)

costvw_fast = nnet.softplus(h_out)
gvw_fast = T.grad(costvw_fast.sum(),filters,consider_constant=[img])
fn_gvw_fast = function([img,filters],gvw_fast)
gvw_fast_value=fn_gvw_fast(img_value,filters_value)

print 'fast_g'
print gvw_fast_value.shape
print gvw_fast_value[0,0,0,:,:]
print gvw_fast_value[1,0,0,:,:]
print gvw_fast_value[0,1,0,:,:]
print gvw_fast_value[1,1,0,:,:]

print 'slow_g'
print gvw_slow_value.shape
print gvw_slow_value[0,0]
print gvw_slow_value[1,0]
print gvw_slow_value[2,0]
print gvw_slow_value[3,0]

"""
TODO test ImgActs
op_recon_img = ImgActs(module_stride)
recon_slow_img =  op_recon_img(filters,vw_slow_sigmoid)
fn_recon_slow_img = theano.function([img,filters],recon_slow_img)
recon_slow_img_value = fn_recon_slow_img(img_value,filters_value)

recon_fast_img = 
"""




