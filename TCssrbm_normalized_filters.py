"""

This file extends the mu-ssRBM for tiled-convolutional training

"""
import cPickle, pickle
import numpy
numpy.seterr('warn') #SHOULD NOT BE IN LIBIMPORT
from PIL import Image
import theano
from theano import tensor
from theano.ifelse import ifelse
from theano.tensor import nnet,grad
from pylearn.io import image_tiling
from pylearn.algorithms.mcRBM import (
        contrastive_cost, contrastive_grad)
import pylearn.gd.sgd

import sys
import os
from unshared_conv_diagonally import FilterActs
from unshared_conv_diagonally import WeightActs
from unshared_conv_diagonally import ImgActs
from Brodatz import Brodatz_op
from Brodatz import Brodatz
from CrossCorrelation import CrossCorrelation,NCC
from MSSIM import MSSIM

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
 	self = cls()
        rng = numpy.random.RandomState(seed)        
        self.s_rng_scan = RandomStreams(int(rng.randint(2**30)))
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

    def gibbs_step_for_v(self, v, s_rng, 
                         border_mask=True, 
                         sampling_for_v=True, 
                         With_fast=True,
                         return_locals=False):
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
            return vv_sample, sample_convhs_s, sample_convhs_h 
        else:
            return vv_sample
        """    
	if return_locals:
            return vv_sample, locals()
        else:
            return vv_sample
        """       
    def gibbs_step_for_v_scan(self, v,border_mask=1,sampling_for_v=1,With_fast=1):
        #positive phase
        s_rng = self.s_rng_scan
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
        cost = cost + self.L1_penalty()
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
    def L1_penalty(self):
        return self.conf['penalty_for_weights_L1'] * abs(self.filters_hs).sum()	
        
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
                '%s_filters_hs.png'%identifier)
   
        if self.conf['lambda_logdomain']:
            raise NotImplementedError()
        else:
	    conv_lambda_for_show = arrange_for_show(self.conv_lambda, self.filters_hs_shape) 	    
	    Image.fromarray(
                            tile_conv_weights(
                            conv_lambda_for_show,flip=False), 'L').save(
                    '%s_conv_lambda.png'%identifier)
     
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
            
        if conf['chain_reset_prob']:
            # advance the 'negative-phase' chain
            nois_batch = self.sampler.s_rng.normal(size=self.rbm.v_shape)
            resets = self.sampler.s_rng.uniform(size=(self.rbm.v_shape[0],))<conf['chain_reset_prob']
            old_particles = tensor.switch(resets.dimshuffle(0,'x','x','x'),
                    nois_batch,   # reset the chain
                    self.sampler.particles,  #continue chain
                    )
            #old_particles = tensor.switch(resets.dimshuffle(0,'x','x','x'),
            #        self.visible_batch,   # reset the chain
            #        self.sampler.particles,  #continue chain
            #        )
        else:
            old_particles = self.sampler.particles
        """
        #cond = theano.printing.Print("cond")(tensor.eq(tensor.floor(self.iteration)%self.conf['reset_interval'],0))
	cond = tensor.eq(tensor.floor(self.iteration)%self.conf['reset_interval'],0)
	steps_sampling = ifelse(cond,
	                              self.conf['constant_steps_sampling'] +self.conf['burn_in'],
	                              self.conf['constant_steps_sampling'])
	#steps_sampling = theano.printing.Print("steps_sampling")(steps_sampling)
	nois_batch = self.sampler.s_rng.normal(size=self.rbm.v_shape)
	nois_batch = tensor.unbroadcast(nois_batch,1) 
	#print nois_batch.broadcastable
	#print type(nois_batch.broadcastable)
	old_particles = ifelse(tensor.eq(tensor.floor(self.iteration)%self.conf['reset_interval'],0),
	                              nois_batch,
	                              self.sampler.particles)             
        """
        tmp_particles = old_particles
        steps_sampling = self.conf['constant_steps_sampling']
        """
        for step in xrange(int(steps_sampling)):
             tmp_particles  = self.rbm.gibbs_step_for_v(tmp_particles,\
                            self.sampler.s_rng,border_mask=conf['border_mask'],\
                            sampling_for_v=conf['sampling_for_v'])
        new_particles = tmp_particles       
        """
        if conf['border_mask']:
	    border_mask_scan = 1
	else:
	    border_mask_scan = 0
	if conf['sampling_for_v']:
	    sampling_for_v_scan = 1
	else:
	    sampling_for_v_scan = 0    
        new_particles, scan_updates=theano.scan(self.rbm.gibbs_step_for_v_scan, 
                                           outputs_info=tmp_particles,
                                           non_sequences=[border_mask_scan,
                                                          sampling_for_v_scan
                                                          ],
                                           n_steps=steps_sampling)
        #print scan_updates
        ups.update(scan_updates)
        
        if conf['normalized_filter']:
	    temp_filters = ups[self.rbm.filters_hs]   
	    for position_index in xrange(self.rbm.filters_hs_shape[0]):
	        for filters_index in xrange(self.rbm.filters_hs_shape[1]):
		    one_filter = temp_filters[position_index,filters_index]
		    one_filter = one_filter/((one_filter**2).sum())
		    dummy = theano.tensor.basic.set_subtensor(temp_filters[position_index,filters_index], one_filter)
	    ups[self.rbm.filters_hs] = temp_filters        
        #reconstructions= self.rbm.gibbs_step_for_v(self.visible_batch, self.sampler.s_rng)
	#recons_error   = tensor.sum((self.visible_batch-reconstructions)**2)
	recons_error = 0.0
        ups[self.recons_error] = recons_error
	#return {self.particles: new_particles}
        ups[self.sampler.particles] = tensor.clip(new_particles[-1],
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
                        
        if self.conf['fast_alpha_min'] < self.conf['alpha_max']:
            if self.conf['alpha_logdomain']:
                ups[self.rbm.conv_alpha_fast] = tensor.clip(
                        ups[self.rbm.conv_alpha_fast],
                        numpy.log(self.conf['fast_alpha_min']).astype(floatX),
                        numpy.log(self.conf['alpha_max']).astype(floatX))
                
            else:
                ups[self.rbm.conv_alpha_fast] = tensor.clip(
                        ups[self.rbm.conv_alpha_fast],
                        self.conf['fast_alpha_min'],
                        self.conf['alpha_max'])
       
        if self.conf['fast_lambda_min'] < self.conf['lambda_max']:
            if self.conf['lambda_logdomain']:
                ups[self.rbm.conv_lambda_fast] = tensor.clip(ups[self.rbm.conv_lambda_fast],
                        numpy.log(self.conf['fast_lambda_min']).astype(floatX),
                        numpy.log(self.conf['lambda_max']).astype(floatX))
                
            else:
                ups[self.rbm.conv_lambda_fast] = tensor.clip(ups[self.rbm.conv_lambda_fast],
                        self.conf['fast_lambda_min'],
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
        pattern = self.conf['directory_name'] + pattern
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
            flip=False),'L').save('%s_particles.png'%pattern)
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

def main_inpaint(filename, 
                 samples_shape=(20,1,76,76),
                 rng=[777888,43,435,678,888], 
                 scale_separately=False, 
                 sampling_for_v=True, 
                 gibbs_steps=501,
                 save_interval=100):
    rbm = cPickle.load(open(filename))
    conf = rbm.conf    
    n_trial = len(rng)
    n_examples, n_img_channels, n_img_rows, n_img_cols=samples_shape
    assert n_img_channels==rbm.v_shape[1]
    assert rbm.filters_hs_shape[-1]==rbm.filters_hs_shape[-2]
    border = rbm.filters_hs_shape[-1]
    assert n_img_rows%border == (border-1)
    assert n_img_cols%border == (border-1)
    rbm.v_shape = (n_examples,n_img_channels,n_img_rows,n_img_cols)
    rbm.out_conv_hs_shape = FilterActs.infer_shape_without_instance(rbm.v_shape,rbm.filters_hs_shape)
    
    B_test = Brodatz([conf['dataset_path']+conf['data_name'],],
                        patch_shape=(n_img_channels,
                                     n_img_rows,
                                     n_img_cols),  
                        noise_concelling=0.0, seed=3322, 
                        batchdata_size=n_examples,
                        rescale=1.0, 
                        new_shapes=[[conf['new_shape_x'],conf['new_shape_x']],],
                        validation=conf['validation'],
                        test_data=True)
                        
    test_shp = B_test.test_img[0].shape
    img = numpy.zeros((1,)+test_shp)    
    img[0,] = B_test.test_img[0]
    temp_img = B_test.test_img[0]
    temp_img_for_save = numpy.asarray(255*(temp_img - temp_img.min()) / (temp_img.max() - temp_img.min() + 1e-6),
                        dtype='uint8')
    Image.fromarray(temp_img_for_save,'L').save('%s_test_img.png'%filename)
    
    inpainted_rows = n_img_rows - 2*border
    inpainted_cols = n_img_cols - 2*border
    
    value_NCC_inpainted_test = numpy.zeros((n_examples*n_trial,))
    value_NCC_inpainted_center = numpy.zeros((n_examples*n_trial,))
    results_f_name = '%s_inapinted.txt'%filename
    results_f = open(results_f_name,'w')
    for n_trial_index in xrange(n_trial):   
        #sampler_rng = numpy.random.RandomState(rng[n_trial_index])
	sampler = Gibbs.alloc(rbm, rng[n_trial_index])
    
	batch_idx = tensor.iscalar()
	batch_range = batch_idx * n_examples + numpy.arange(n_examples)
     
	batch_x = Brodatz_op(batch_range,
  	                     [conf['dataset_path']+conf['data_name'],],  # download from http://www.ux.uis.no/~tranden/brodatz.html
  	                     patch_shape=(n_img_channels,
  	                                 n_img_rows,
  	                                 n_img_cols), 
  	                     noise_concelling=0., 
  	                     seed=rng[n_trial_index], 
  	                     batchdata_size=n_examples,
  	                     rescale=1.0,
                             new_shapes=[[conf['new_shape_x'],conf['new_shape_x']],],
                             validation=conf['validation'],
                             test_data=True #we use test image
  	                     )	
	fn_getdata = theano.function([batch_idx],batch_x)
	batchdata = fn_getdata(0)
	scaled_batchdata_center = numpy.zeros((n_examples,n_img_channels,inpainted_rows,inpainted_cols))
	scaled_batchdata_center[:,:,:,:] = batchdata[:,:,border:n_img_rows-border,border:n_img_cols-border]   
    
        batchdata[:,:,border:n_img_rows-border,border:n_img_cols-border] = 0
        print 'the min of border: %f, the max of border: %f'%(batchdata.min(),batchdata.max())
        shared_batchdata = sharedX(batchdata,'batchdata')
	border_mask = numpy.zeros((n_examples,n_img_channels,n_img_rows,n_img_cols),dtype=floatX)
	border_mask[:,:,border:n_img_rows-border,border:n_img_cols-border]=1
        
	sampler.particles = shared_batchdata
	new_particles = rbm.gibbs_step_for_v(sampler.particles, 
	                                     sampler.s_rng,
	                                     border_mask=False, 
                                             sampling_for_v=sampling_for_v,
                                             With_fast=False)
	new_particles = tensor.mul(new_particles,border_mask)
	new_particles = tensor.add(new_particles,batchdata)
	fn = theano.function([], [],
                updates={sampler.particles: new_particles})
	#particles = sampler.particles

	    
        savename = '%s_inpaint_%i_trail_%i_%04i.png'%(filename,n_img_rows,n_trial_index,0)
        temp = sampler.particles.get_value(borrow=True)
        Image.fromarray(
                tile_conv_weights(
                                 temp,
                                 flip=False,scale_each=True
                                 ),
                        'L').save(savename)
	for i in xrange(gibbs_steps):
	    #print i
	    if i % save_interval == 0 and i != 0:
		savename = '%s_inpaint_%i_trail_%i_%04i.png'%(filename,n_img_rows,n_trial_index,i)
		print 'saving'
		mean_particles = rbm.gibbs_step_for_v(
                        sampler.particles, 
                        sampler.s_rng,
                        border_mask=False, 
                        sampling_for_v=False,
                        With_fast=False)
                mean_particles = tensor.mul(mean_particles,border_mask)
		mean_particles = tensor.add(mean_particles,batchdata)                        
		fn_generate_mean = theano.function([],mean_particles)  
		samples_for_show = fn_generate_mean()
		print 'the min of center: %f, the max of center: %f' \
                                 %(samples_for_show[:,:,border:n_img_rows-border,border:n_img_cols-border].min(),
                                 samples_for_show[:,:,border:n_img_rows-border,border:n_img_cols-border].max())
                          
		if scale_separately:
		    pass
		    """
		    scale_separately_savename = '%s_inpaint_scale_separately_%04i.png'%(filename,i)
		    blank_img = numpy.zeros((n_examples,n_img_channels,n_img_rows,n_img_cols),dtype=floatX)
		    tmp = temp[:,:,11:66,11:66]
		    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6)
		    blank_img[:,:,11:66,11:66] = tmp 
		    blank_img = blank_img + scaled_batchdata
		    Image.fromarray(
		    tile_conv_weights(
			blank_img,
			flip=False,scale_each=True),
		    'L').save(scale_separately_savename)
		    """
		else:
		    Image.fromarray(
			tile_conv_weights(
			samples_for_show,
			flip=False,scale_each=True),
			'L').save(savename)
		    tmp = samples_for_show
		    inpainted_img = tmp[:,:,border:n_img_rows-border,border:n_img_cols-border]
		    #print inpainted_img.shape
		    #print scaled_batchdata_center.shape
		    value_NCC = NCC(scaled_batchdata_center, inpainted_img)	
		    print i
		    print 'NCC'
		    print '%f, %f'%(value_NCC.mean(),value_NCC.std())
		    results_f.write('trail %i, %04i\n'%(n_trial_index,i))
		    results_f.write('NCC\n')
		    results_f.write('%f, %f\n'%(value_NCC.mean(),value_NCC.std()))
		    
		    _,_,rows,cols = inpainted_img.shape
		    assert rows==cols
		    CC = CrossCorrelation(img,inpainted_img,
			  window_size=rows, n_patches_of_samples=0)
		    print img.shape
		    print inpainted_img.shape
		    value_TSS =  CC.TSS()
		    print 'TSS'
		    print '%f, %f'%(value_TSS.mean(),value_TSS.std())
		    results_f.write('TSS\n')
		    results_f.write('%f, %f\n'%(value_TSS.mean(),value_TSS.std()))   
		    center_MSSIM = 255*(scaled_batchdata_center - scaled_batchdata_center.min())/(scaled_batchdata_center.max()-scaled_batchdata_center.min()+1e-6)
		    inpainted_MSSIM = 255*(inpainted_img - scaled_batchdata_center.min())/(scaled_batchdata_center.max()-scaled_batchdata_center.min()+1e-6)
		    mssim = MSSIM(center_MSSIM, inpainted_MSSIM,11)
		    mssim_mean, mssim_std = mssim.MSSIM()
		    print 'MSSIM score : %f, %f\n'%(mssim_mean, mssim_std)
		    results_f.write('MSSIM score\n')
		    results_f.write('%f, %f\n'%(mssim_mean, mssim_std))
	    fn()
	start = n_trial_index*n_examples
	end = (n_trial_index+1)*n_examples
        value_NCC_inpainted_center[start:end,] = value_NCC
        value_NCC_inpainted_test[start:end,] = value_TSS
    
    results_f.write('Final NCC\n')
    results_f.write('%f, %f\n'%(value_NCC_inpainted_center.mean(),value_NCC_inpainted_center.std()))    
    results_f.write('Final TSS\n')
    results_f.write('%f, %f\n'%(value_NCC_inpainted_test.mean(),value_NCC_inpainted_test.std())) 
    results_f.close()
    
def main_sample(filename,
                samples_shape=(128,1,120,120),
                burn_in=10001,
                save_interval=1000, 
                sampling_for_v=True, 
                rng=777888):
    rbm = cPickle.load(open(filename))
    filename = filename + '_watch_sh'
    conf = rbm.conf
    n_samples, n_channels, n_img_rows, n_img_cols = samples_shape
    assert n_channels==rbm.v_shape[1]
    rbm.v_shape = (n_samples, n_channels, n_img_rows, n_img_cols)
    assert rbm.filters_hs_shape[-1]==rbm.filters_hs_shape[-2]
    border = rbm.filters_hs_shape[-1]
    assert n_img_rows%border == (border-1)
    assert n_img_cols%border == (border-1)
    rbm.out_conv_hs_shape = FilterActs.infer_shape_without_instance(rbm.v_shape,rbm.filters_hs_shape)    
    n_total_hidden_units = 1
    for h_shape_index in xrange(len(rbm.out_conv_hs_shape)):
        n_total_hidden_units = n_total_hidden_units*rbm.out_conv_hs_shape[h_shape_index]
    rbm.out_conv_hs_shape[0]*rbm.out_conv_hs_shape[1]*rbm.out_conv_hs_shape[2]
    s_particles = sharedX(numpy.zeros(rbm.out_conv_hs_shape),'s')
    h_particles = sharedX(numpy.zeros(rbm.out_conv_hs_shape),'h')
    sampler = Gibbs.alloc(rbm, rng)
    new_particles, new_s_particles, new_h_particles = rbm.gibbs_step_for_v(
						    sampler.particles, 
						    sampler.s_rng,
						    border_mask=False, 
						    sampling_for_v=sampling_for_v,
						    With_fast=False,
						    return_locals=True)
    """                    
    new_particles = tensor.clip(new_particles,
                rbm.conf['particles_min'],
                rbm.conf['particles_max'])
    """
    #if we need clip the generated samples, the model is wrong
    fn = theano.function([], [],
                updates={sampler.particles: new_particles,
                         s_particles: new_s_particles,
                         h_particles: new_h_particles})
    particles = sampler.particles
    B_texture = Brodatz([conf['dataset_path']+conf['data_name'],],
                        #patch_shape=samples_shape[1:], 
                        patch_shape=(1,98,98),
                        noise_concelling=0.0, 
                        seed=rng, 
                        batchdata_size=1, 
                        rescale=1.0,
                        #new_shapes=[[480,480],],
                        new_shapes=[[conf['new_shape_x'],conf['new_shape_x']],],
                        validation=conf['validation'],
                        test_data=False)
    
    test_shp = B_texture.test_img[0].shape
    img = numpy.zeros((1,)+test_shp)    
    img[0,] = B_texture.test_img[0]
    temp_img = B_texture.test_img[0]
    temp_img_for_save = numpy.asarray(255*(temp_img - temp_img.min()) / (temp_img.max() - temp_img.min() + 1e-6),
                        dtype='uint8')
    Image.fromarray(temp_img_for_save,'L').save('%s_test_img.png'%filename)
        
    results_f_name = '%s_sample_%i_TCC.txt'%(filename,n_img_rows)
    results_f = open(results_f_name,'w')
    savename = '%s_sample_%i_burn_0.png'%(filename,n_img_rows)
    Image.fromarray(
                tile_conv_weights(
                                 sampler.particles.get_value(borrow=True)[:,:,border:n_img_rows-border,border:n_img_cols-border],
                                 flip=False,scale_each=True
                                 ),
                        'L').save(savename)
    for i in xrange(burn_in):
	if i% 40 ==0:
	    print i	
	    temp_s = s_particles.get_value(borrow=True)
	    temp_h = h_particles.get_value(borrow=True)
	    #temp_sh = temp_s*temp_h
	    #min_s = temp_sh.min()
	    #max_s = temp_sh.max()
	    min_s = temp_s.min()
	    max_s = temp_s.max()
	    results_f.write('%04i\n'%i)
	    results_f.write('s, (%f,%f)\n'%(min_s,max_s))
	    print 's, (%f,%f)'%(min_s,max_s)
	    h_sparsity = temp_h.sum()/n_total_hidden_units
	    print 'the sparsity of h: %f'%h_sparsity
	    results_f.write('the sparsity of h: %f\n'%h_sparsity)
        savename = '%s_sample_%i_burn_%04i.png'%(filename,n_img_rows,i)
	if i % save_interval == 0 and i != 0:
	    print 'saving'
	    mean_particles = rbm.gibbs_step_for_v(
                        sampler.particles, 
                        sampler.s_rng,
                        border_mask=False, 
                        sampling_for_v=False,
                        With_fast=False)
            fn_generate_mean = theano.function([],mean_particles)  
            samples_for_show = fn_generate_mean()
            #import pdb;pdb.set_trace()
            Image.fromarray(
                tile_conv_weights(
                                 samples_for_show[:,:,border:n_img_rows-border,border:n_img_cols-border],
                                 flip=False,scale_each=True
                                 ),
                        'L').save(savename)	
            samples = samples_for_show[:,:,border:n_img_rows-border,border:n_img_cols-border]
            CC = CrossCorrelation(img,samples,
                       window_size=19, n_patches_of_samples=1)
	    aaa = CC.TSS()
	    print aaa.mean(),aaa.std()
	    results_f.write('%f, %f\n'%(aaa.mean(),aaa.std()))
        fn()       
    results_f.close()   
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
"""     

def main_sampling_inpaint(filename):
    main_sample(filename)    
    main_inpaint(filename)
    main_sample(filename,
                samples_shape=(1,1,362,362),
                burn_in=10001,
                save_interval=1000, 
                sampling_for_v=True, 
                rng=777888)
                
def main0(conf):
    
    batchsize = conf['batchsize']

    batch_idx = tensor.iscalar()
    batch_range = batch_idx * conf['batchsize'] + numpy.arange(conf['batchsize'])
    
    
       
    n_examples = conf['batchsize']   #64
    n_img_rows = 98
    n_img_cols = 98
    n_img_channels=1
    batch_x = Brodatz_op(batch_range,
  	                 [conf['dataset_path']+conf['data_name'],],   # download from http://www.ux.uis.no/~tranden/brodatz.html
  	                 patch_shape=(n_img_channels,
  	                              n_img_rows,
  	                              n_img_cols), 
  	                 noise_concelling=0., 
  	                 seed=3322, 
  	                 batchdata_size=n_examples,
                         rescale=1.0,
                         new_shapes=[[conf['new_shape_x'],conf['new_shape_x']],],
                         validation=conf['validation'],
                         test_data=False
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

    rbm.save_weights_to_grey_files(conf['directory_name']+'iter_0000')

    base_lr = conf['base_lr_per_example']/batchsize
    conv_lr_coef = conf['conv_lr_coef']
    if conf['FPCD']:
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
                rbm.conv_lambda: sharedX(conv_lr_coef*0.0, 'conv_lambda_lr'),
                rbm.v_prec_fast: sharedX(base_lr, 'prec_lr_fast'),
                rbm.filters_hs_fast: sharedX(conv_lr_coef*base_lr, 'filters_hs_lr_fast'),
                rbm.conv_bias_hs_fast: sharedX(base_lr, 'conv_bias_hs_lr_fast'),
                rbm.conv_mu_fast: sharedX(base_lr, 'conv_mu_lr_fast'),
                rbm.conv_alpha_fast: sharedX(base_lr, 'conv_alpha_lr_fast'),
                rbm.conv_lambda_fast: sharedX(conv_lr_coef*0.0, 'conv_lambda_lr_fast'),
                },
            conf = conf,
            )
    else:
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
                rbm.conv_lambda: sharedX(conv_lr_coef*0.0, 'conv_lambda_lr'),
                rbm.v_prec_fast: sharedX(0.0, 'prec_lr_fast'),
                rbm.filters_hs_fast: sharedX(conv_lr_coef*0.0, 'filters_hs_lr_fast'),
                rbm.conv_bias_hs_fast: sharedX(0.0, 'conv_bias_hs_lr_fast'),
                rbm.conv_mu_fast: sharedX(0.0, 'conv_mu_lr_fast'),
                rbm.conv_alpha_fast: sharedX(0.0, 'conv_alpha_lr_fast'),
                rbm.conv_lambda_fast: sharedX(conv_lr_coef*0.0, 'conv_lambda_lr_fast'),
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
        if iter % 40 == 0:
	    trainer.print_status()	    
	if iter % 10000 == 0:
            rbm.dump_to_file(conf['directory_name']+'rbm_%06i.pkl'%iter)
        if iter <= 1000 and not (iter % 100): #
            trainer.print_status()
            trainer.save_weights_to_grey_files()
        elif not (iter % 1000):
            trainer.print_status()
            trainer.save_weights_to_grey_files()
        iter += 1   

def main_train(argv):
    print 'start main_train'
    print argv
    conf=dict(
            dataset_path='/data/lisa/exp/luoheng/Brodatz/',
            #data_rescale = [2,], #2 (4.0/3) means rescale images from 640*640 to 320*320 
            #chain_reset_prob=0.005,#reset for approximately every 200 iterations
            #reset_interval=500,
            burn_in=20,
            unnatural_grad=False,
            #alpha_logdomain=True,
            #conv_alpha0=100.,
            alpha_min=1.010102,
            fast_alpha_min=0.99,
            alpha_max=1000.,
            lambda_min=0.,
            fast_lambda_min=-0.01,
            lambda_max=10.,
            lambda0=0.0,
            lambda_logdomain=False,
            conv_bias0=0.0, 
            conv_bias_irange=0.0,#conv_bias0 +- this
            conv_mu0 = 1.0,
            #train_iters=40000,
            #base_lr_per_example=0.00001,
            conv_lr_coef=1.0,
            batchsize=64,
            n_filters_hs=32,
            #v_prec_init=10., # this should increase with n_filters_hs?
            #v_prec_lower_limit = 10.01,
            v_prec_fast_lower_limit = -0.01,
	    filters_hs_size=11,
            filters_irange=.01,
            #sparsity_weight_conv=0,#numpy.float32(500),
            #sparsity_weight_global=0.,
            particles_min=-1000.,
            particles_max=1000.,
            #problem_term_vWWv_weight = 0.,
            #problem_term_vIv_weight = 0.,
            #constant_steps_sampling = 5,         
            #increase_steps_sampling = True,
            border_mask=True,
            sampling_for_v=True,
            #penalty_for_weights_L1=0.002,
            #penalty_for_fast_parameters = 0.05,
            #FPCD = False,
            )
    directory_name = ''        
    conf['data_name'] = argv[2] #D6.gif    
    directory_name += conf['data_name']
    directory_name += '_'
    
    conf['new_shape_x'] = int(argv[3]) #320,480,640
    directory_name = directory_name + 'new_shape_x' + argv[3]
    directory_name += '_'
    
    if int(argv[4])==1:
        conf['validation'] = True
        directory_name += 'validation'
        directory_name += '_'
    else: 
        conf['validation'] = False
        directory_name += 'test'
        directory_name += '_'
    
    if int(argv[5])==1:
        conf['FPCD'] = True
        directory_name += 'FPCD'
        directory_name += '_'
    else: 
        conf['FPCD'] = False
        directory_name += 'PCD' 
        directory_name += '_'
    """
    conf['reset_interval'] = int(argv[6])
    directory_name = directory_name + 'reset' + argv[6]
    directory_name += '_'
    """
    conf['chain_reset_prob'] = float(argv[6])
    directory_name = directory_name + 'reset' + argv[6]
    directory_name += '_'
    
    if int(argv[7])==1:
        conf['alpha_logdomain'] = True        
    else:
        conf['alpha_logdomain'] = False
    directory_name += 'alpha'    
    directory_name += argv[7]
    directory_name += '_'
    
    conf['conv_alpha0'] = float(argv[8])
    directory_name += argv[8]
    directory_name += '_'
    
    conf['train_iters'] = int(argv[9])
    directory_name += 'epoch'
    directory_name += argv[9]
    directory_name += '_'
    
    conf['base_lr_per_example'] = float(argv[10])
    directory_name += 'lr'
    directory_name += argv[10]
    directory_name += '_'
    
    conf['constant_steps_sampling'] = int(argv[11])
    directory_name += 'gibbs'
    directory_name += argv[11]
    directory_name += '_'
    
    conf['penalty_for_fast_parameters'] = float(argv[12])
    directory_name += 'fast_weight_decay'
    directory_name += argv[12]
    directory_name += '_'
    
    conf['penalty_for_weights_L1'] = float(argv[13])
    directory_name += 'L1_penalty'
    directory_name += argv[13]
    directory_name += '_'
    
    conf['v_prec_init'] = float(argv[14])
    directory_name += 'v_prec_init'
    directory_name += argv[14]
    directory_name += '_'
    
    conf['v_prec_lower_limit'] = float(argv[15])
    directory_name += 'v_prec_lower_limit'
    directory_name += argv[15]
    directory_name += '_'
   
    if int(argv[16])==1:
        conf['normalized_filter'] = True        
    else:
        conf['normalized_filter'] = False
    directory_name += 'normalize_filter'    
    directory_name += argv[16]
       
    directory_name += '/'    
    os.mkdir(directory_name)
    conf['directory_name'] = directory_name
    print 'conf'
    print conf
    main0(conf)   
    
    rbm_file = conf['directory_name'] + 'rbm_%06i.pkl'%conf['train_iters']
    main_sample(rbm_file)    
    main_inpaint(rbm_file)
    main_sample(rbm_file,
                samples_shape=(1,1,362,362),
                burn_in=10001,
                save_interval=1000, 
                sampling_for_v=True, 
                rng=777888)

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        sys.exit(main_train(sys.argv))
    if sys.argv[1] == 'sampling':
	sys.exit(main_sample(sys.argv[2]))
    if sys.argv[1] == 'inpaint':
        sys.exit(main_inpaint(sys.argv[2]))
    if sys.argv[1] == 's_i':
        sys.exit(main_sampling_inpaint(sys.argv[2]))