"""

This file extends the mu-ssRBM for tiled-convolutional training

"""
import cPickle, pickle
import numpy
numpy.seterr('warn') #SHOULD NOT BE IN LIBIMPORT
from PIL import Image
import theano
from theano import tensor
from theano.tensor import nnet,grad
from pylearn.io import image_tiling
from pylearn.algorithms.mcRBM import (
        contrastive_cost, contrastive_grad)
import pylearn.gd.sgd

import sys
from unshared_conv_diagonally import FilterActs
from unshared_conv_diagonally import WeightActs
from unshared_conv_diagonally import ImgActs
from Brodatz import Brodatz_op
from Brodatz import Brodatz
from CrossCorrelation import CrossCorrelation

#import scipy.io
import os
_temp_data_path_ = '.'#'/Tmp/luoheng'

if 1:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


floatX=theano.config.floatX
sharedX = lambda X, name : theano.shared(numpy.asarray(X, dtype=floatX),
        name=name)

def Toncv(image,filters,module_stride=1):
    op = FilterActs(module_stride)
    return op(image,filters)
    
def Tdeconv(filters, hidacts, irows, icols, module_stride=1):
    op = ImgActs(module_stride)
    return op(filters, hidacts, irows, icols)


def unnatural_sgd_updates(params, grads, stepsizes, tracking_coef=0.1, epsilon=1):
    grad_means = [theano.shared(numpy.zeros_like(p.get_value(borrow=True)))
            for p in params]
    grad_means_sqr = [theano.shared(numpy.ones_like(p.get_value(borrow=True)))
            for p in params]
    updates = dict()
    for g, gm, gms, p, s in zip(
            grads, grad_means, grad_means_sqr, params, stepsizes):
        updates[gm] = tracking_coef * g + (1-tracking_coef) * gm
        updates[gms] = tracking_coef * g*g + (1-tracking_coef) * gms

        var_g = gms - gm**2
        # natural grad doesn't want sqrt, but i found it worked worse
        updates[p] = p - s * gm / tensor.sqrt(var_g+epsilon)
    return updates
"""
def grad_updates(params, grads, stepsizes):
    grad_means = [theano.shared(numpy.zeros_like(p.get_value(borrow=True)))
            for p in params]
    grad_means_sqr = [theano.shared(numpy.ones_like(p.get_value(borrow=True)))
            for p in params]
    updates = dict()
    for g, p, s in zip(
            grads, params, stepsizes):
        updates[p] = p - s*g
    return updates
"""
def safe_update(a, b):
    for k,v in dict(b).iteritems():
        if k in a:
            raise KeyError(k)
        a[k] = v
    return a
    
def most_square_shape(N):
    """rectangle (height, width) with area N that is closest to sqaure
    """
    for i in xrange(int(numpy.sqrt(N)),0, -1):
        if 0 == N % i:
            return (i, N/i)


def tile_conv_weights(w,flip=False, scale_each=False):
    """
    Return something that can be rendered as an image to visualize the filters.
    """
    #if w.shape[1] != 3:
    #    raise NotImplementedError('not rgb', w.shape)
    if w.shape[2] != w.shape[3]:
        raise NotImplementedError('not square', w.shape)

    if w.shape[1] == 1:
	wmin, wmax = w.min(), w.max()
    	if not scale_each:
            w = numpy.asarray(255 * (w - wmin) / (wmax - wmin + 1e-6), dtype='uint8')
    	trows, tcols= most_square_shape(w.shape[0])
    	outrows = trows * w.shape[2] + trows-1
    	outcols = tcols * w.shape[3] + tcols-1
    	out = numpy.zeros((outrows, outcols), dtype='uint8')
    	#tr_stride= 1+w.shape[1]
    	for tr in range(trows):
            for tc in range(tcols):
            	# this is supposed to flip the filters back into the image
            	# coordinates as well as put the channels in the right place, but I
            	# don't know if it really does that
            	tmp = w[tr*tcols+tc,
			     0,
                             ::-1 if flip else 1,
                             ::-1 if flip else 1]
            	if scale_each:
                    tmp = numpy.asarray(255*(tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6),
                        dtype='uint8')
            	out[tr*(1+w.shape[2]):tr*(1+w.shape[2])+w.shape[2],
                    tc*(1+w.shape[3]):tc*(1+w.shape[3])+w.shape[3]] = tmp
    	return out

    wmin, wmax = w.min(), w.max()
    if not scale_each:
        w = numpy.asarray(255 * (w - wmin) / (wmax - wmin + 1e-6), dtype='uint8')
    trows, tcols= most_square_shape(w.shape[0])
    outrows = trows * w.shape[2] + trows-1
    outcols = tcols * w.shape[3] + tcols-1
    out = numpy.zeros((outrows, outcols,3), dtype='uint8')

    tr_stride= 1+w.shape[1]
    for tr in range(trows):
        for tc in range(tcols):
            # this is supposed to flip the filters back into the image
            # coordinates as well as put the channels in the right place, but I
            # don't know if it really does that
            tmp = w[tr*tcols+tc].transpose(1,2,0)[
                             ::-1 if flip else 1,
                             ::-1 if flip else 1]
            if scale_each:
                tmp = numpy.asarray(255*(tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6),
                        dtype='uint8')
            out[tr*(1+w.shape[2]):tr*(1+w.shape[2])+w.shape[2],
                    tc*(1+w.shape[3]):tc*(1+w.shape[3])+w.shape[3]] = tmp
    return out

class RBM(object):
    """
    Light-weight class that provides math related to inference in Spike & Slab RBM

    Attributes:
     - v_prec - the base conditional precisions of data units [shape (n_img_rows, n_img_cols,)]
     - v_shape - the input image shape  (ie. n_imgs, n_chnls, n_img_rows, n_img_cols)

     - n_conv_hs - the number of spike and slab hidden units
     - filters_hs_shape - the kernel filterbank shape for hs units
     - filters_h_shape -  the kernel filterbank shape for h units
     - filters_hs - a tensor with shape (n_conv_hs,n_chnls,n_ker_rows, n_ker_cols)
     - conv_bias_hs - a vector with shape (n_conv_hs, n_out_rows, n_out_cols)
     - subsample_hs - how to space the receptive fields (dx,dy)

     - n_global_hs - how many globally-connected spike and slab units
     - weights_hs - global weights
     - global_bias_hs -

     - _params a list of the attributes that are shared vars


    The technique of combining convolutional and global filters to account for border effects is
    borrowed from  (Alex Krizhevsky, TR?, October 2010).
    """
    def __init__(self, **kwargs):
        print 'init rbm'
	self.__dict__.update(kwargs)

    @classmethod
    def alloc(cls,
            conf,
            image_shape,  # input dimensionality
            filters_hs_shape,       
            filters_irange,
            v_prec,
            v_prec_lower_limit, #should be parameter of the training algo            
            seed = 8923402            
            ):
 	print 'alloc rbm'
        rng = numpy.random.RandomState(seed)

        self = cls()
       
	n_images, n_channels, n_img_rows, n_img_cols = image_shape
        n_filters_hs_modules, n_filters_hs_per_modules, fcolors, n_filters_hs_rows, n_filters_hs_cols = filters_hs_shape        
        assert fcolors == n_channels        
        self.v_shape = image_shape
        print 'v_shape'
	print self.v_shape
	self.filters_hs_shape = filters_hs_shape
        print 'self.filters_hs_shape'
        print self.filters_hs_shape
        self.out_conv_hs_shape = FilterActs.infer_shape_without_instance(self.v_shape,self.filters_hs_shape)        
        print 'self.out_conv_hs_shape'
        print self.out_conv_hs_shape
        #conv_bias_hs_shape = self.out_conv_hs_shape[1:]
        conv_bias_hs_shape = (n_filters_hs_modules, n_filters_hs_per_modules) 
        self.conv_bias_hs_shape = conv_bias_hs_shape
        print 'self.conv_bias_hs_shape'
        print self.conv_bias_hs_shape
        #self.v_prec = sharedX(numpy.zeros((n_channels, n_img_rows, n_img_cols))+v_prec, 'var_v_prec')
        #self.v_prec_fast = sharedX(numpy.zeros((n_channels, n_img_rows, n_img_cols))+conf['v_prec_lower_limit'], 'var_v_prec_fast')
        self.v_prec = sharedX(v_prec, 'var_v_prec')        
        self.v_prec_fast = sharedX(0.0, 'var_v_prec_fast')        
        self.v_prec_lower_limit = sharedX(v_prec_lower_limit, 'v_prec_lower_limit')
        self.v_prec_fast_lower_limit = sharedX(conf['v_prec_fast_lower_limit'], 'v_prec_fast_lower_limit')
        
        self.filters_hs = sharedX(rng.randn(*filters_hs_shape) * filters_irange , 'filters_hs')  
        self.filters_hs_fast = sharedX(numpy.zeros(filters_hs_shape), 'filters_hs_fast') 
        
	self.conv_bias_hs = sharedX(numpy.zeros(self.conv_bias_hs_shape), name='conv_bias_hs')
        self.conv_bias_hs_fast = sharedX(numpy.zeros(self.conv_bias_hs_shape), name='conv_bias_hs_fast')
        
        conv_mu_ival = numpy.zeros(conv_bias_hs_shape,dtype=floatX) + conf['conv_mu0']
	self.conv_mu = sharedX(conv_mu_ival, name='conv_mu')
	self.conv_mu_fast = sharedX(numpy.zeros(conv_bias_hs_shape,dtype=floatX), name='conv_mu_fast')
        
	if conf['alpha_logdomain']:
            conv_alpha_ival = numpy.zeros(conv_bias_hs_shape,dtype=floatX) + numpy.log(conf['conv_alpha0'])
	    self.conv_alpha = sharedX(conv_alpha_ival,'conv_alpha')
	    conv_alpha_ival_fast = numpy.zeros(conv_bias_hs_shape,dtype=floatX)
	    #conv_alpha_ival_fast = numpy.zeros(conv_bias_hs_shape,dtype=floatX) + numpy.log(conf['alpha_min'])
	    self.conv_alpha_fast = sharedX(conv_alpha_ival_fast, name='conv_alpha_fast')
	else:
            self.conv_alpha = sharedX(
                    numpy.zeros(conv_bias_hs_shape)+conf['conv_alpha0'],
                    'conv_alpha')
            self.conv_alpha_fast = sharedX(
                    numpy.zeros(conv_bias_hs_shape), name='conv_alpha_fast')        
            #self.conv_alpha_fast = sharedX(
            #        numpy.zeros(conv_bias_hs_shape)+conf['alpha_min'], name='conv_alpha_fast')        
 
        if conf['lambda_logdomain']:
            self.conv_lambda = sharedX(
                    numpy.zeros(self.filters_hs_shape)
                        + numpy.log(conf['lambda0']),
                    name='conv_lambda')
            self.conv_lambda_fast = sharedX(
                    numpy.zeros(self.filters_hs_shape),
                    name='conv_lambda_fast')        
            #self.conv_lambda_fast = sharedX(
            #        numpy.zeros(self.filters_hs_shape)
            #            + numpy.log(conf['lambda_min']),
            #        name='conv_lambda_fast')         
        else:
            self.conv_lambda = sharedX(
                    numpy.zeros(self.filters_hs_shape)
                        + (conf['lambda0']),
                    name='conv_lambda')
            self.conv_lambda_fast = sharedX(
                    numpy.zeros(self.filters_hs_shape),
                    name='conv_lambda_fast')        
            #self.conv_lambda_fast = sharedX(
            #        numpy.zeros(self.filters_hs_shape)
            #            + (conf['lambda_min']),
            #        name='conv_lambda_fast')        

        negsample_mask = numpy.zeros((n_channels,n_img_rows,n_img_cols),dtype=floatX)
 	negsample_mask[:,n_filters_hs_rows:n_img_rows-n_filters_hs_rows+1,n_filters_hs_cols:n_img_cols-n_filters_hs_cols+1] = 1
	self.negsample_mask = sharedX(negsample_mask,'negsample_mask')                
        
        self.conf = conf
        self._params = [self.v_prec,
                self.filters_hs,
                self.conv_bias_hs,
                self.conv_mu, 
                self.conv_alpha,
                self.conv_lambda
                ]
        self._params_fast = [self.v_prec_fast,
                self.filters_hs_fast,
                self.conv_bias_hs_fast,
                self.conv_mu_fast,
                self.conv_alpha_fast,
                self.conv_lambda_fast
	        ]
                    
        return self

    def get_conv_alpha(self,With_fast):
        if With_fast:
	    if self.conf['alpha_logdomain']:
                rval = tensor.exp(self.conv_alpha+self.conv_alpha_fast)
	        return rval
            else:
                return self.conv_alpha+self.conv_alpha_fast
        else:
	    if self.conf['alpha_logdomain']:
                rval = tensor.exp(self.conv_alpha)
	        return rval
            else:
                return self.conv_alpha
                
    def get_conv_lambda(self,With_fast):
        if With_fast:
	    if self.conf["lambda_logdomain"]:
                L = tensor.exp(self.conv_lambda+self.conv_lambda_fast)
            else:
                L = self.conv_lambda+self.conv_lambda_fast
        else:
	    if self.conf["lambda_logdomain"]:
                L = tensor.exp(self.conv_lambda)
            else:
                L = self.conv_lambda
        return L
    def get_v_prec(self,With_fast):
        if With_fast:
	    return self.v_prec+self.v_prec_fast
	else:
	    return self.v_prec	    
    def get_filters_hs(self,With_fast):
        if With_fast:
	    return self.filters_hs+self.filters_hs_fast
	else:
	    return self.filters_hs
    def get_conv_bias_hs(self,With_fast):
        if With_fast:
	    return self.conv_bias_hs+self.conv_bias_hs_fast
	else:
	    return self.conv_bias_hs
    def get_conv_mu(self,With_fast):
        if With_fast:
	    return self.conv_mu+self.conv_mu_fast
	else:
	    return self.conv_mu
        
    def conv_problem_term(self, v, With_fast):
        L = self.get_conv_lambda(With_fast)
        vLv = self.convdot(v*v, L)        
        return vLv
    def conv_problem_term_T(self, h, With_fast):
        L = self.get_conv_lambda(With_fast)
        #W = self.filters_hs
        #alpha = self.get_conv_alpha()
        hL = self.convdot_T(L, h)        
        return hL
    def convdot(self, image, filters):
        return Toncv(image,filters)
        
    def convdot_T(self, filters, hidacts):
        n_images, n_channels, n_img_rows, n_img_cols = self.v_shape
        return Tdeconv(filters, hidacts, n_img_rows, n_img_cols)         

    #####################
    # spike-and-slab convolutional hidden units
    def mean_convhs_h_given_v(self, v, With_fast):
        """Return the mean of binary-valued hidden units h, given v
        """
        alpha = self.get_conv_alpha(With_fast)
        W = self.get_filters_hs(With_fast)
        vW = self.convdot(v, W)
        vW_broadcastable = vW.dimshuffle(0,3,4,1,2)
        #change 64 x 11 x 32 x 8 x 8 to 64 x 8 x 8 x 11 x 32 for broadcasting
        pre_convhs_h_parts = self.get_conv_mu(With_fast)*vW_broadcastable + self.get_conv_bias_hs(With_fast) +  0.5*(vW_broadcastable**2)/alpha
        rval = nnet.sigmoid(
                tensor.add(
                    pre_convhs_h_parts.dimshuffle(0,3,4,1,2),
                    -0.5*self.conv_problem_term(v,With_fast)))
        return rval

    def mean_var_convhs_s_given_v(self, v, With_fast):
        """
        Return mu (N,K,B) and sigma (N,K,K) for latent s variable.

        For efficiency, this method assumes all h variables are 1.

        """
        alpha = self.get_conv_alpha(With_fast)
        W = self.get_filters_hs(With_fast)
        vW = self.convdot(v, W)
        rval = self.get_conv_mu(With_fast) + (vW.dimshuffle(0,3,4,1,2))/alpha        
        return rval.dimshuffle(0,3,4,1,2), 1.0 / alpha

    #####################
    # visible units
    def mean_var_v_given_h_s(self, convhs_h, convhs_s, With_fast):
        shF = self.convdot_T(self.get_filters_hs(With_fast), convhs_h*convhs_s)        
        conv_hL = self.conv_problem_term_T(convhs_h,With_fast)
        contrib = shF               
        sigma_sq = 1.0 / (self.get_v_prec(With_fast) + conv_hL)
        mu = contrib * sigma_sq        
        return mu, sigma_sq


    def all_hidden_h_means_given_v(self, v, With_fast):
        mean_convhs_h = self.mean_convhs_h_given_v(v,With_fast)
        return mean_convhs_h

    #####################

    def gibbs_step_for_v(self, v, s_rng, return_locals=False, border_mask=True, sampling_for_v=True, With_fast=True):
        #positive phase

        # spike variable means
        mean_convhs_h = self.all_hidden_h_means_given_v(v,With_fast)
        #broadcastable_value = mean_convhs_h.broadcastable
        #print broadcastable_value
        
        # slab variable means
        meanvar_convhs_s = self.mean_var_convhs_s_given_v(v,With_fast)
        #smean, svar = meanvar_convhs_s
        #broadcastable_value = smean.broadcastable
        #print broadcastable_value
        #broadcastable_value = svar.broadcastable
        #print broadcastable_value
        
        # spike variable samples
        def sample_h(hmean,shp):
            return tensor.cast(s_rng.uniform(size=shp) < hmean, floatX)
        #def sample_s(smeanvar, shp):
        #    smean, svar = smeanvar
        #    return s_rng.normal(size=shp)*tensor.sqrt(svar) + smean

        sample_convhs_h = sample_h(mean_convhs_h, self.out_conv_hs_shape)
        
        # slab variable samples
        smean, svar = meanvar_convhs_s 
        # the shape of svar: n_filters_hs_modules, n_filters_hs_per_modules
        random_normal = s_rng.normal(size=self.out_conv_hs_shape)
        random_normal_bc = random_normal.dimshuffle(0,3,4,1,2)*tensor.sqrt(svar)
        sample_convhs_s = random_normal_bc.dimshuffle(0,3,4,1,2) + smean
        	
        #negative phase
        vv_mean, vv_var = self.mean_var_v_given_h_s(
                sample_convhs_h, sample_convhs_s,With_fast
                )
        if sampling_for_v:
	    vv_sample = s_rng.normal(size=self.v_shape) * tensor.sqrt(vv_var) + vv_mean
	else:
	    vv_sample = vv_mean 
        if border_mask:
	    vv_sample = theano.tensor.mul(vv_sample,self.negsample_mask)
        #broadcastable_value = vv_mean.broadcastable
        #print broadcastable_value
       
	if return_locals:
            return vv_sample, locals()
        else:
            return vv_sample
               
   
    def free_energy_given_v(self, v, With_fast=False):
        # This is accurate up to a multiplicative constant
        # because I dropped some terms involving 2pi
        alpha = self.get_conv_alpha(With_fast)	    
        W = self.get_filters_hs(With_fast)        
        vW = self.convdot(v, W)
        vW_broadcastable = vW.dimshuffle(0,3,4,1,2)
        #change 64 x 11 x 32 x 8 x 8 to 64 x 8 x 8 x 11 x 32 for broadcasting
        pre_convhs_h_parts = self.get_conv_mu(With_fast)*vW_broadcastable + self.get_conv_bias_hs(With_fast) +  0.5*(vW_broadcastable**2)/alpha
                
	pre_convhs_h = tensor.add(
                    pre_convhs_h_parts.dimshuffle(0,3,4,1,2),
                   -0.5*self.conv_problem_term(v,With_fast))
        rval = tensor.add(
                -tensor.sum(nnet.softplus(pre_convhs_h),axis=[1,2,3,4]), #the shape of pre_convhs_h: 64 x 11 x 32 x 8 x 8
                0.5 * tensor.sum(self.get_v_prec(With_fast) * (v**2), axis=[1,2,3]), #shape: 64 x 1 x 98 x 98 
                )
        assert rval.ndim==1
        return rval
        

    def cd_updates(self, pos_v, neg_v, stepsizes, other_cost=None):      
        cost=(self.free_energy_given_v(pos_v,With_fast=False) - self.free_energy_given_v(neg_v,With_fast=False)).sum()
        if other_cost:
	    cost = cost + other_cost
	grads = theano.tensor.grad(cost,
		#wrt=self.params()+self.params_fast(),
		wrt=self.params(),
		consider_constant=[pos_v]+[neg_v])
        
        print len(stepsizes),len(grads+grads)
        assert len(stepsizes)==len(grads+grads)

        if self.conf['unnatural_grad']:
            sgd_updates = unnatural_sgd_updates
        else:
            sgd_updates = pylearn.gd.sgd.sgd_updates
        rval = dict(
                sgd_updates(
                    self.params()+self.params_fast(),
                    grads+grads,
                    stepsizes=stepsizes))
        """            
        if 0:
            #DEBUG STORE GRADS
            grad_shared_vars = [sharedX(0*p.value.copy(),'') for p in self.params()]
            self.grad_shared_vars = grad_shared_vars
            rval.update(dict(zip(grad_shared_vars, grads)))
        """
	return rval

    def params(self):
        # return the list of *shared* learnable parameters
        # that are, in your judgement, typically learned in this model
        return list(self._params)
    def params_fast(self):
        # return the list of *shared* learnable parameters
        # that are, in your judgement, typically learned in this model
        return list(self._params_fast)    

    def save_weights_to_files(self, identifier):
        # save 4 sets of weights:
        pass
    def save_weights_to_grey_files(self, identifier):
        # save 4 sets of weights:

        #filters_hs
        def arrange_for_show(filters_hs,filters_hs_shape):
	    n_filters_hs_modules, n_filters_hs_per_modules, fcolors, n_filters_hs_rows, n_filters_hs_cols  = filters_hs_shape            
            filters_fs_for_show = filters_hs.reshape(
                       (n_filters_hs_modules*n_filters_hs_per_modules, 
                       fcolors,
                       n_filters_hs_rows,
                       n_filters_hs_cols))
            fn = theano.function([],filters_fs_for_show)
            rval = fn()
            return rval
        filters_fs_for_show = arrange_for_show(self.filters_hs, self.filters_hs_shape)
        Image.fromarray(
                       tile_conv_weights(
                       filters_fs_for_show,flip=False), 'L').save(
                'filters_hs_%s.png'%identifier)
   
        if self.conf['lambda_logdomain']:
            raise NotImplementedError()
        else:
	    conv_lambda_for_show = arrange_for_show(self.conv_lambda, self.filters_hs_shape) 	    
	    Image.fromarray(
                            tile_conv_weights(
                            conv_lambda_for_show,flip=False), 'L').save(
                    'conv_lambda_%s.png'%identifier)
     
    def dump_to_file(self, filename):
        try:
            cPickle.dump(self, open(filename, 'wb'))
        except cPickle.PicklingError:
            pickle.dump(self, open(filename, 'wb'))


class Gibbs(object): # if there's a Sampler interface - this should support it
    @classmethod
    def alloc(cls, rbm, rng):
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self = cls()
        seed=int(rng.randint(2**30))
        self.rbm = rbm
        self.particles = sharedX(
            rng.randn(*rbm.v_shape),
            name='particles')
	#self.particles = sharedX(
        #    numpy.zeros(rbm.v_shape),
        #    name='particles')
	self.s_rng = RandomStreams(seed)
        return self

def HMC(rbm, batchsize, rng): # if there's a Sampler interface - this should support it
    if not hasattr(rng, 'randn'):
        rng = numpy.random.RandomState(rng)
    seed=int(rng.randint(2**30))
    particles = sharedX(
            rng.randn(*rbm.v_shape),
            name='particles')
    return pylearn.sampling.hmc.HMC_sampler(
            particles,
            rbm.free_energy_given_v,
            seed=seed)


class Trainer(object): # updates of this object implement training
    @classmethod
    def alloc(cls, rbm, visible_batch,
            lrdict,
            conf,
            rng=234,
            iteration_value=0,
            ):

        batchsize = rbm.v_shape[0]
        sampler = Gibbs.alloc(rbm, rng=rng)
	print 'alloc trainer'
        error = 0.0
        return cls(
                rbm=rbm,
                batchsize=batchsize,
                visible_batch=visible_batch,
                sampler=sampler,
                iteration=sharedX(iteration_value, 'iter'), #float32.....
                learn_rates = [lrdict[p] for p in rbm.params()],
                learn_rates_fast = [lrdict[p_fast] for p_fast in rbm.params_fast()],
                conf=conf,
                annealing_coef=sharedX(1.0, 'annealing_coef'),
                conv_h_means = sharedX(numpy.zeros(rbm.out_conv_hs_shape[1:])+0.5,'conv_h_means'),
                cpnv_h       = sharedX(numpy.zeros(rbm.out_conv_hs_shape), 'conv_h'),
                recons_error = sharedX(error,'reconstruction_error'),                
                )

    def __init__(self, **kwargs):
        print 'init trainer'
	self.__dict__.update(kwargs)

    def updates(self):
        
        print 'start trainer.updates'
	conf = self.conf
        ups = {}
        add_updates = lambda b: safe_update(ups,b)

        annealing_coef = 1.0 - self.iteration / float(conf['train_iters'])
        ups[self.iteration] = self.iteration + 1 #
        ups[self.annealing_coef] = annealing_coef

        conv_h = self.rbm.all_hidden_h_means_given_v(
                self.visible_batch, With_fast=True)
        
        
        new_conv_h_means = 0.1 * conv_h.mean(axis=0) + .9*self.conv_h_means
        #new_conv_h_means = conv_h.mean(axis=0)
        ups[self.conv_h_means] = new_conv_h_means
        ups[self.cpnv_h] = conv_h
        #ups[self.global_h_means] = new_global_h_means


        #sparsity_cost = 0
        #self.sparsity_cost = sparsity_cost
        # SML updates PCD
        add_updates(
                self.rbm.cd_updates(
                    pos_v=self.visible_batch,
                    neg_v=self.sampler.particles,
                    stepsizes=[annealing_coef*lr for lr in self.learn_rates]+[lr_fast for lr_fast in self.learn_rates_fast]))
        if conf['increase_steps_sampling']:
	    steps_sampling = self.iteration.get_value() / 1000 + self.conf['constant_steps_sampling']
	else:
	    steps_sampling = self.conf['constant_steps_sampling']
	    
        if conf['chain_reset_prob']:
            # advance the 'negative-phase' chain
            nois_batch = self.sampler.s_rng.normal(size=self.rbm.v_shape)
            #steps_sampling = steps_sampling + conf['chain_reset_burn_in']
            resets = self.sampler.s_rng.uniform()<conf['chain_reset_prob']
            old_particles = tensor.switch(resets.dimshuffle('x','x','x','x'),
                    nois_batch,   # reset the chain
                    self.sampler.particles,  #continue chain
                    )
            #old_particles = tensor.switch(resets.dimshuffle(0,'x','x','x'),
            #        self.visible_batch,   # reset the chain
            #        self.sampler.particles,  #continue chain
            #        )
        else:
            old_particles = self.sampler.particles
        
	#print steps_sampling        
        tmp_particles = old_particles    
        for step in xrange(int(steps_sampling)):
             tmp_particles  = self.rbm.gibbs_step_for_v(tmp_particles,\
                            self.sampler.s_rng,border_mask=conf['border_mask'],\
                            sampling_for_v=conf['sampling_for_v'])
        new_particles = tmp_particles       
        #broadcastable_value = new_particles.broadcastable
        #print broadcastable_value
        #reconstructions= self.rbm.gibbs_step_for_v(self.visible_batch, self.sampler.s_rng)
	#recons_error   = tensor.sum((self.visible_batch-reconstructions)**2)
	recons_error = 0.0
        ups[self.recons_error] = recons_error
	#return {self.particles: new_particles}
        ups[self.sampler.particles] = tensor.clip(new_particles,
                conf['particles_min'],
                conf['particles_max'])
        
        
        
        # make sure that the new v_precision doesn't top below its floor
        new_v_prec = ups[self.rbm.v_prec]
        ups[self.rbm.v_prec] = tensor.switch(
                new_v_prec<self.rbm.v_prec_lower_limit,
                self.rbm.v_prec_lower_limit,
                new_v_prec)
        
        if self.conf['alpha_min'] < self.conf['alpha_max']:
            if self.conf['alpha_logdomain']:
                ups[self.rbm.conv_alpha] = tensor.clip(
                        ups[self.rbm.conv_alpha],
                        numpy.log(self.conf['alpha_min']).astype(floatX),
                        numpy.log(self.conf['alpha_max']).astype(floatX))
                
            else:
                ups[self.rbm.conv_alpha] = tensor.clip(
                        ups[self.rbm.conv_alpha],
                        self.conf['alpha_min'],
                        self.conf['alpha_max'])
       
        if self.conf['lambda_min'] < self.conf['lambda_max']:
            if self.conf['lambda_logdomain']:
                ups[self.rbm.conv_lambda] = tensor.clip(ups[self.rbm.conv_lambda],
                        numpy.log(self.conf['lambda_min']).astype(floatX),
                        numpy.log(self.conf['lambda_max']).astype(floatX))
                
            else:
                ups[self.rbm.conv_lambda] = tensor.clip(ups[self.rbm.conv_lambda],
                        self.conf['lambda_min'],
                        self.conf['lambda_max'])
       
        weight_decay = numpy.asarray(self.conf['penalty_for_fast_parameters'], dtype=floatX)
        
        for p_fast in self.rbm.params_fast():
	    new_p_fast = ups[p_fast]
	    new_p_fast = new_p_fast - weight_decay*p_fast
	    ups[p_fast] = new_p_fast
	           
        new_v_prec_fast = ups[self.rbm.v_prec_fast]        
        ups[self.rbm.v_prec_fast] = tensor.switch(
                new_v_prec_fast<self.rbm.v_prec_fast_lower_limit,
                self.rbm.v_prec_fast_lower_limit,
                new_v_prec_fast)
                        
        if self.conf['alpha_min'] < self.conf['alpha_max']:
            if self.conf['alpha_logdomain']:
                ups[self.rbm.conv_alpha_fast] = tensor.clip(
                        ups[self.rbm.conv_alpha_fast],
                        numpy.log(self.conf['alpha_min']).astype(floatX),
                        numpy.log(self.conf['alpha_max']).astype(floatX))
                
            else:
                ups[self.rbm.conv_alpha_fast] = tensor.clip(
                        ups[self.rbm.conv_alpha_fast],
                        self.conf['alpha_min'],
                        self.conf['alpha_max'])
       
        if self.conf['lambda_min'] < self.conf['lambda_max']:
            if self.conf['lambda_logdomain']:
                ups[self.rbm.conv_lambda_fast] = tensor.clip(ups[self.rbm.conv_lambda_fast],
                        numpy.log(self.conf['lambda_min']).astype(floatX),
                        numpy.log(self.conf['lambda_max']).astype(floatX))
                
            else:
                ups[self.rbm.conv_lambda_fast] = tensor.clip(ups[self.rbm.conv_lambda_fast],
                        self.conf['lambda_min'],
                        self.conf['lambda_max'])                
                        
                        
        return ups

    def save_weights_to_files(self, pattern='iter_%05i'):
        #pattern = pattern%self.iteration.get_value()

        # save particles
        #Image.fromarray(tile_conv_weights(self.sampler.particles.get_value(borrow=True),
        #    flip=False),
        #        'RGB').save('particles_%s.png'%pattern)
        #self.rbm.save_weights_to_files(pattern)
        pass

    def save_weights_to_grey_files(self, pattern='iter_%05i'):
        pattern = pattern%self.iteration.get_value()

        # save particles
        """
        particles_for_show = self.sampler.particles.dimshuffle(3,0,1,2)
        fn = theano.function([],particles_for_show)
        particles_for_show_value = fn()
        Image.fromarray(tile_conv_weights(particles_for_show_value,
            flip=False),'L').save('particles_%s.png'%pattern)
        self.rbm.save_weights_to_grey_files(pattern)
        """
        Image.fromarray(tile_conv_weights(self.sampler.particles.get_value(borrow=True),
            flip=False),'L').save('particles_%s.png'%pattern)
        self.rbm.save_weights_to_grey_files(pattern)
    def print_status(self):
        def print_minmax(msg, x):
            assert numpy.all(numpy.isfinite(x))
            print msg, x.min(), x.max()

        print 'iter:', self.iteration.get_value()
        print_minmax('filters_hs', self.rbm.filters_hs.get_value(borrow=True))
        print_minmax('filters_hs_fast', self.rbm.filters_hs_fast.get_value(borrow=True))
        print_minmax('conv_bias_hs', self.rbm.conv_bias_hs.get_value(borrow=True))
        print_minmax('conv_bias_hs_fast', self.rbm.conv_bias_hs_fast.get_value(borrow=True))
        print_minmax('conv_mu', self.rbm.conv_mu.get_value(borrow=True))
        print_minmax('conv_mu_fast', self.rbm.conv_mu_fast.get_value(borrow=True))
        if self.conf['alpha_logdomain']:
            print_minmax('conv_alpha',
                    numpy.exp(self.rbm.conv_alpha.get_value(borrow=True)))
            print_minmax('conv_alpha_fast',
                    numpy.exp(self.rbm.conv_alpha_fast.get_value(borrow=True)))
        else:
            print_minmax('conv_alpha', self.rbm.conv_alpha.get_value(borrow=True))
            print_minmax('conv_alpha_fast', self.rbm.conv_alpha_fast.get_value(borrow=True))
        if self.conf['lambda_logdomain']:
            print_minmax('conv_lambda',
                    numpy.exp(self.rbm.conv_lambda.get_value(borrow=True)))
            print_minmax('conv_lambda_fast',
                    numpy.exp(self.rbm.conv_lambda_fast.get_value(borrow=True)))
        else:
            print_minmax('conv_lambda', self.rbm.conv_lambda.get_value(borrow=True))
            print_minmax('conv_lambda_fast', self.rbm.conv_lambda_fast.get_value(borrow=True))
        print_minmax('v_prec', self.rbm.v_prec.get_value(borrow=True))
        print_minmax('v_prec_fast', self.rbm.v_prec_fast.get_value(borrow=True))
        print_minmax('particles', self.sampler.particles.get_value())
        print_minmax('conv_h_means', self.conv_h_means.get_value())
        print_minmax('conv_h', self.cpnv_h.get_value())
        print (self.cpnv_h.get_value()).std()
        #print self.conv_h_means.get_value()[0,0:11,0:11]
	#print self.rbm.conv_bias_hs.get_value(borrow=True)[0,0,0:3,0:3]
        #print self.rbm.h_tiled_conv_mask.get_value(borrow=True)[0,32,0:3,0:3]
	#print_minmax('global_h_means', self.global_h_means.get_value())
        print 'lr annealing coef:', self.annealing_coef.get_value()
	#print 'reconstruction error:', self.recons_error.get_value()

def main_inpaint(filename, algo='Gibbs', rng=777888, scale_separately=False, sampling_for_v=False):
    rbm = cPickle.load(open(filename))
    sampler = Gibbs.alloc(rbm, rng)
    
    batch_idx = tensor.iscalar()
    batch_range = batch_idx * rbm.conf['batchsize'] + numpy.arange(rbm.conf['batchsize'])
    
    n_examples = rbm.conf['batchsize']   #64
    n_img_rows = 98
    n_img_cols = 98
    n_img_channels=1
    batch_x = Brodatz_op(batch_range,
  	                     '../../Brodatz/D6.gif',   # download from http://www.ux.uis.no/~tranden/brodatz.html
  	                     patch_shape=(n_img_channels,
  	                                 n_img_rows,
  	                                 n_img_cols), 
  	                     noise_concelling=0., 
  	                     seed=3322, 
  	                     batchdata_size=n_examples
  	                     )	
    fn_getdata = theano.function([batch_idx],batch_x)
    batchdata = fn_getdata(0)
    scaled_batchdata = (batchdata - batchdata.min())/(batchdata.max() - batchdata.min() + 1e-6)
    scaled_batchdata[:,:,11:88,11:88] = 0
    
    batchdata[:,:,11:88,11:88] = 0
    print 'the min of border: %f, the max of border: %f'%(batchdata.min(),batchdata.max())
    shared_batchdata = sharedX(batchdata,'batchdata')
    border_mask = numpy.zeros((n_examples,n_img_channels,n_img_rows,n_img_cols),dtype=floatX)
    border_mask[:,:,11:88,11:88]=1
        
    sampler.particles = shared_batchdata
    new_particles = rbm.gibbs_step_for_v(sampler.particles, sampler.s_rng, 
                                 sampling_for_v=sampling_for_v,With_fast=False)
    new_particles = tensor.mul(new_particles,border_mask)
    new_particles = tensor.add(new_particles,batchdata)
    fn = theano.function([], [],
                updates={sampler.particles: new_particles})
    particles = sampler.particles


    for i in xrange(5000):
        print i
        if i % 100 == 0:
            savename = '%s_inpaint_%04i.png'%(filename,i)
            print 'saving'
            temp = particles.get_value(borrow=True)
            print 'the min of center: %f, the max of center: %f' \
                                 %(temp[:,:,11:88,11:88].min(),temp[:,:,11:88,11:88].max())
            if scale_separately:
	        scale_separately_savename = '%s_inpaint_scale_separately_%04i.png'%(filename,i)
	        blank_img = numpy.zeros((n_examples,n_img_channels,n_img_rows,n_img_cols),dtype=floatX)
	        tmp = temp[:,:,11:88,11:88]
	        tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6)
	        blank_img[:,:,11:88,11:88] = tmp 
	        blank_img = blank_img + scaled_batchdata
	        Image.fromarray(
                tile_conv_weights(
                    blank_img,
                    flip=False,scale_each=True),
                'L').save(scale_separately_savename)
            else:
	        Image.fromarray(
                tile_conv_weights(
                    particles.get_value(borrow=True),
                    flip=False,scale_each=True),
                'L').save(savename)
        fn()

def main_sample(filename, algo='Gibbs', rng=777888, burn_in=50001, save_interval=5000, n_files=10, sampling_for_v=False):
    rbm = cPickle.load(open(filename))
    n_samples = 128
    rbm.v_shape = (n_samples,1,120,120)
    rbm.out_conv_hs_shape = FilterActs.infer_shape_without_instance(rbm.v_shape,rbm.filters_hs_shape)
    rbm.v_prec = sharedX(numpy.zeros(rbm.v_shape[1:])+rbm.v_prec.get_value(borrow=True).mean(), 'var_v_prec')
    if algo == 'Gibbs':
        sampler = Gibbs.alloc(rbm, rng)
        new_particles  = rbm.gibbs_step_for_v(
                        sampler.particles, sampler.s_rng,
                        border_mask=False, sampling_for_v=sampling_for_v,
                        With_fast=False)
        new_particles = tensor.clip(new_particles,
                rbm.conf['particles_min'],
                rbm.conf['particles_max'])
        fn = theano.function([], [],
                updates={sampler.particles: new_particles})
        particles = sampler.particles
    elif algo == 'HMC':
        print "WARNING THIS PROBABLY DOESNT WORK"
        # still need to figure out how to get the clipping into
        # the iterations of mcmc
        sampler = HMC(rbm, rbm.conf['batchsize'], rng)
        ups = sampler.updates()
        ups[sampler.positions] = tensor.clip(ups[sampler.positions],
                rbm.conf['particles_min'],
                rbm.conf['particles_max'])
        fn = theano.function([], [], updates=ups)
        particles = sampler.positions
    
    B_texture = Brodatz('../../../Brodatz/D6.gif', patch_shape=(1,98,98), 
                         noise_concelling=0.0, seed=3322 ,batchdata_size=1, rescale=1.0, rescale_size=2)
    shp = B_texture.test_img.shape
    img = numpy.zeros((1,)+shp)
    temp_img = numpy.asarray(B_texture.test_img, dtype='uint8')
    img[0,] = temp_img
    Image.fromarray(temp_img,'L').save('test_img.png')    
    for i in xrange(burn_in):
	if i% 100 ==0:
	    print i	
        #savename = '%s_Large_sample_burn_%04i.png'%(filename,i)        	
	#tmp = particles.get_value(borrow=True)[0,0,11:363,11:363]
	#w = numpy.asarray(255 * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6), dtype='uint8')
	#Image.fromarray(w,'L').save(savename)		
	savename = '%s_sample_burn_%04i.png'%(filename,i)
	if i % 1000 == 0 and i!=0:
	    print 'saving'
            Image.fromarray(
                tile_conv_weights(
                    particles.get_value(borrow=True)[:,:,11:110,11:110],
                    flip=False,scale_each=True),
                'L').save(savename)	
            samples = particles.get_value(borrow=True)[:,:,11:110,11:110]
            for samples_index in xrange(n_samples):
                temp_samples = samples[samples_index,]
                #temp_samples = numpy.asarray(255 * (temp_samples - temp_samples.min()) / \
                #                   (temp_samples.max() - temp_samples.min() + 1e-6), dtype='uint8')
                samples[samples_index,]= temp_samples
            CC = CrossCorrelation(img,samples,
                       window_size=19, n_patches_of_samples=1)
	    aaa = CC.TSS()
	    print aaa.mean(),aaa.std()
        fn()   
    
    """
    for n in xrange(n_files):
        for i in xrange(save_interval):
            fn()
        savename = '%s_sample_%04i.png'%(filename,n)
        print 'saving', savename
        Image.fromarray(
                tile_conv_weights(
                    particles.get_value(borrow=True),
                    flip=False,scale_each=True),
                'L').save(savename)
    """            
def main_print_status(filename, algo='Gibbs', rng=777888, burn_in=500, save_interval=500, n_files=1):
    def print_minmax(msg, x):
        assert numpy.all(numpy.isfinite(x))
        print msg, x.min(), x.max()
    rbm = cPickle.load(open(filename))
    if algo == 'Gibbs':
        sampler = Gibbs.alloc(rbm, rng)
        new_particles  = rbm.gibbs_step_for_v(sampler.particles, sampler.s_rng)
        #new_particles = tensor.clip(new_particles,
        #        rbm.conf['particles_min'],
        #        rbm.conf['particles_max'])
        fn = theano.function([], [],
                updates={sampler.particles: new_particles})
        particles = sampler.particles
    elif algo == 'HMC':
        print "WARNING THIS PROBABLY DOESNT WORK"
     
    for i in xrange(burn_in):
	fn()
        print_minmax('particles', particles.get_value(borrow=True))              
                
                
def main0(rval_doc):
    if 'conf' not in rval_doc:
        raise NotImplementedError()

    conf = rval_doc['conf']
    batchsize = conf['batchsize']

    batch_idx = tensor.iscalar()
    batch_range = batch_idx * conf['batchsize'] + numpy.arange(conf['batchsize'])
    
    
       
    n_examples = conf['batchsize']   #64
    n_img_rows = 98
    n_img_cols = 98
    n_img_channels=1
    batch_x = Brodatz_op(batch_range,
  	                     conf['dataset'],   # download from http://www.ux.uis.no/~tranden/brodatz.html
  	                     patch_shape=(n_img_channels,
  	                                 n_img_rows,
  	                                 n_img_cols), 
  	                     noise_concelling=0., 
  	                     seed=3322, 
  	                     batchdata_size=n_examples,
                             rescale=1.0,
                             rescale_size=conf['data_rescale']
  	                     )	
           
    rbm = RBM.alloc(
            conf,
            image_shape=(        
                n_examples,
                n_img_channels,
                n_img_rows,
                n_img_cols
                ),
            filters_hs_shape=(
                conf['filters_hs_size'],  
                conf['n_filters_hs'],
                n_img_channels,
                conf['filters_hs_size'],
                conf['filters_hs_size']
                ),            #fmodules(stride) x filters_per_modules x fcolors(channels) x frows x fcols
            filters_irange=conf['filters_irange'],
            v_prec=conf['v_prec_init'],
            v_prec_lower_limit=conf['v_prec_lower_limit'],            
            )

    rbm.save_weights_to_grey_files('iter_0000')

    base_lr = conf['base_lr_per_example']/batchsize
    conv_lr_coef = conf['conv_lr_coef']

    trainer = Trainer.alloc(
            rbm,
            visible_batch=batch_x,
            lrdict={
                # higher learning rate ok with CD1
                rbm.v_prec: sharedX(base_lr, 'prec_lr'),
                rbm.filters_hs: sharedX(conv_lr_coef*base_lr, 'filters_hs_lr'),
                rbm.conv_bias_hs: sharedX(base_lr, 'conv_bias_hs_lr'),
                rbm.conv_mu: sharedX(base_lr, 'conv_mu_lr'),
                rbm.conv_alpha: sharedX(base_lr, 'conv_alpha_lr'),
                rbm.conv_lambda: sharedX(conv_lr_coef*base_lr, 'conv_lambda_lr'),
                rbm.v_prec_fast: sharedX(base_lr, 'prec_lr_fast'),
                rbm.filters_hs_fast: sharedX(conv_lr_coef*base_lr, 'filters_hs_lr_fast'),
                rbm.conv_bias_hs_fast: sharedX(base_lr, 'conv_bias_hs_lr_fast'),
                rbm.conv_mu_fast: sharedX(base_lr, 'conv_mu_lr_fast'),
                rbm.conv_alpha_fast: sharedX(base_lr, 'conv_alpha_lr_fast'),
                rbm.conv_lambda_fast: sharedX(conv_lr_coef*base_lr, 'conv_lambda_lr_fast'),
                },
            conf = conf,
            )

  
    print 'start building function'
    training_updates = trainer.updates() #
    train_fn = theano.function(inputs=[batch_idx],
            outputs=[],
	    #mode='FAST_COMPILE',
            #mode='DEBUG_MODE',
	    updates=training_updates	    
	    )  #

    print 'training...'
    
    iter = 0
    while trainer.annealing_coef.get_value()>=0: #
        dummy = train_fn(iter) #
        if iter % 10 == 0:
	    trainer.print_status()
	if iter % 1000 == 0:
            rbm.dump_to_file(os.path.join(_temp_data_path_,'rbm_%06i.pkl'%iter))
        if iter <= 1000 and not (iter % 100): #
            trainer.print_status()
            trainer.save_weights_to_grey_files()
        elif not (iter % 1000):
            trainer.print_status()
            trainer.save_weights_to_grey_files()
        iter += 1



def main_train():
    print 'start main_train'
    main0(dict(
        conf=dict(
            dataset='../../Brodatz/D6.gif',
            data_rescale = 2, #2 (4.0/3) means rescale images from 640*640 to 320*320 
            chain_reset_prob=0.001,#reset for approximately every 1000 iterations
            #chain_reset_iterations=1000,
            chain_reset_burn_in=20,
            unnatural_grad=False,
            alpha_logdomain=True,
            conv_alpha0=100.,
            global_alpha0=10.,
            alpha_min=1.,
            alpha_max=1000.,
            lambda_min=0.,
            lambda_max=10.,
            lambda0=0.001,
            lambda_logdomain=False,
            conv_bias0=0.0, 
            conv_bias_irange=0.0,#conv_bias0 +- this
            conv_mu0 = 1.0,
            train_iters=40000,
            base_lr_per_example=0.00001,
            conv_lr_coef=1.0,
            batchsize=64,
            n_filters_hs=32,
            v_prec_init=10., # this should increase with n_filters_hs?
            v_prec_lower_limit = 10.,
            v_prec_fast_lower_limit = 0.,
	    filters_hs_size=11,
            filters_irange=.01,
            zero_out_interior_weights=False,
            #sparsity_weight_conv=0,#numpy.float32(500),
            #sparsity_weight_global=0.,
            particles_min=-1000.,
            particles_max=1000.,
            #problem_term_vWWv_weight = 0.,
            #problem_term_vIv_weight = 0.,
            n_tiled_conv_offset_diagonally = 1,
            constant_steps_sampling = 5,         
            increase_steps_sampling = True,
            border_mask=True,
            sampling_for_v=True,
            penalty_for_fast_parameters = 0.05
            )))
    

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        sys.exit(main_train())
    if sys.argv[1] == 'sampling':
	sys.exit(main_sample(sys.argv[2]))
    if sys.argv[1] == 'inpaint':
        sys.exit(main_inpaint(sys.argv[2]))
    if sys.argv[1] == 'print_status':
        sys.exit(main_print_status(sys.argv[2]))
