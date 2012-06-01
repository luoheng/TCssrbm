"""

This file implements the binary convolutional ssRBM as a second layer in DBN

"""
import cPickle, pickle
import numpy
numpy.seterr('warn') #SHOULD NOT BE IN LIBIMPORT
from PIL import Image
import theano
from theano import tensor
from theano.tensor import nnet,grad
from theano.tensor.nnet.conv import conv2d
from pylearn.io import image_tiling
from pylearn.algorithms.mcRBM import (
        contrastive_cost, contrastive_grad)
import pylearn.gd.sgd
from TCssrbm_FPCD import RBM,Gibbs

import sys
#from unshared_conv_diagonally import FilterActs
#from unshared_conv_diagonally import WeightActs
#from unshared_conv_diagonally import ImgActs
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

def conv2d_transpose(x, filters, in_img_shape, filters_shape, subsample):
    """
    Supposing a linear transformation M implementing convolution by dot(img, M),
    Return the equivalent of dot(x, M.T).

    This is also implemented by a convolution, but with lots of dimshuffles and flipping and
    stuff.
    """
    dummy_v = tensor.tensor4()
    z_hs = conv2d(dummy_v, filters,
            image_shape=in_img_shape,
            filter_shape=filters_shape,
            subsample=subsample)
    rval, _ = z_hs.owner.op.grad((dummy_v, filters), (x,))
    return rval

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


def tile_conv_weights(w,flip=False, scale_each=True):
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

class bRBM(object):
    """
    Light-weight class that provides math related to inference in binary Spike & Slab RBM

    Attributes:
         - _params a list of the attributes that are shared vars
    """
    def __init__(self, **kwargs):
        print 'init binary rbm'
	self.__dict__.update(kwargs)

    @classmethod
    def alloc(cls,
            l2_conf,
            hs_shape,  # input dimensionality
            filters_shape,       
            filters_irange,
            rbm,
            seed = 8923402,            
            ):
 	print 'alloc rbm'
        rng = numpy.random.RandomState(seed)

        self = cls()
        #print hs_shape
        #print filters_shape
	n_batchsize, n_maps_, n_hs_rows, n_hs_cols = hs_shape
        n_filters, n_maps, n_filters_rows, n_filters_cols = filters_shape        
        assert n_maps_ == n_maps        
        self.hs_shape = hs_shape
        print 'hs_shape'
	print self.hs_shape	
	self.filters_shape = filters_shape
        print 'self.filters_shape'
        print self.filters_shape
        self.out_conv_v_shape = (n_batchsize, n_filters, n_hs_rows-n_filters_rows+1, n_hs_cols-n_filters_cols+1)        
        print 'self.out_conv_v_shape'
        print self.out_conv_v_shape
        
        #start to define the parameters
        #biases for v and h
        conv_v_bias_shape = self.out_conv_v_shape[1:]         
        self.conv_v_bias_shape = conv_v_bias_shape
        self.conv_v_bias = sharedX(numpy.zeros(self.conv_v_bias_shape), name='conv_v_bias')
        self.conv_v_bias_fast = sharedX(numpy.zeros(self.conv_v_bias_shape), name='conv_v_bias_fast')
        print 'self.conv_v_bias_shape'
        print self.conv_v_bias_shape
        
        h_bias_shape = self.hs_shape[1:]
        self.h_bias_shape = h_bias_shape
        
        def conver_hs_bias(a,old_shp=rbm.conv_bias_hs_shape,new_shp=self.h_bias_shape):
	    f_modules,n_filters = old_shp
	    n_maps, n_hs_rows, n_hs_cols = new_shp
	    assert f_modules*n_filters == n_maps
	    b = a.reshape(f_modules*n_filters)
	    rval = numpy.zeros(new_shp)
	    for filters_index in xrange(f_modules*n_filters):
		rval[filters_index,:,:]= b[filters_index]
	    return rval
	    
        h_bias_ival = conver_hs_bias(rbm.conv_bias_hs.get_value())
        self.h_bias = sharedX(h_bias_ival, 'h_bias')
        #self.h_bias = sharedX(numpy.zeros(self.h_bias_shape), 'h_bias')        
        self.h_bias_fast = sharedX(numpy.zeros(self.h_bias_shape), 'h_bias_fast')     
        print 'self.h_bias_shape'
        print self.h_bias_shape                  
        
        #filters
        self.filters = sharedX(rng.randn(*self.filters_shape) * filters_irange , 'filters_hs')  
        self.filters_fast = sharedX(numpy.zeros(filters_shape), 'filters_fast')        
        	
	#mu        
        mu_shape = self.hs_shape[1:]
        self.mu_shape = mu_shape
        #mu_ival = numpy.zeros(mu_shape,dtype=floatX) + l2_conf['mu0']
	mu_ival = conver_hs_bias(rbm.conv_mu.get_value())
	self.mu = sharedX(mu_ival, name='mu')
	self.mu_fast = sharedX(numpy.zeros(mu_shape,dtype=floatX), name='mu_fast')
        print 'mu_shape'
        print self.mu_shape
        
        if l2_conf['alpha_logdomain']:
            #alpha_ival = numpy.zeros(self.mu_shape,dtype=floatX) + numpy.log(l2_conf['alpha0'])
	    alpha_ival = conver_hs_bias(rbm.conv_alpha.get_value())
	    self.alpha = sharedX(alpha_ival,'alpha')
	    alpha_ival_fast = numpy.zeros(self.mu_shape,dtype=floatX)
	    self.alpha_fast = sharedX(alpha_ival_fast, name='alpha_fast')
	else:
            alpha_ival = conver_hs_bias(rbm.conv_alpha.get_value())
            self.alpha = sharedX(
                    alpha_ival,
                    'alpha')
            self.alpha_fast = sharedX(
                    numpy.zeros(self.mu_shape), name='alpha_fast')            
        
        self.l2_conf = l2_conf
        self._params = [self.filters,
                self.conv_v_bias,
                self.h_bias,
                self.mu, 
                self.alpha               
                ]
        self._params_fast = [self.filters_fast,
                self.conv_v_bias_fast,
                self.h_bias_fast,
                self.mu_fast,
                self.alpha_fast
	        ]                    
        return self
    def get_filters(self,With_fast):
        if With_fast:
	    return self.filters+self.filters_fast
	else:
	    return self.filters
    def get_alpha(self,With_fast):
        if With_fast:
	    if self.l2_conf['alpha_logdomain']:
                rval = tensor.exp(self.alpha+self.alpha_fast)
	        return rval
            else:
                return self.alpha+self.alpha_fast
        else:
	    if self.l2_conf['alpha_logdomain']:
                rval = tensor.exp(self.alpha)
	        return rval
            else:
                return self.alpha
    def get_conv_v_bias(self,With_fast):
        if With_fast:
	    return self.conv_v_bias+self.conv_v_bias_fast
	else:
	    return self.conv_v_bias
    def get_h_bias(self,With_fast):
        if With_fast:
	    return self.h_bias+self.h_bias_fast
	else:
	    return self.h_bias	    
    def get_mu(self,With_fast):
        if With_fast:
	    return self.mu+self.mu_fast
	else:
	    return self.mu

    def convdot(self,hs,filters):
        return conv2d(hs,filters,
                image_shape=self.hs_shape,
                filter_shape=self.filters_shape,
                subsample=(1,1))
    def convdot_T(self, v, filters):
        return conv2d_transpose(v, filters,
                   self.hs_shape,
                   self.filters_shape,
                   (1,1))
    #####################
    # binary spike-and-slab convolutional visible units
    def mean_conv_v_given_s_h(self, s, h, With_fast):
        """Return the mean of binary-valued visible units v, given h and s
        """
        W = self.get_filters(With_fast)
        conv_v_bias = self.get_conv_v_bias(With_fast)
        shW = self.convdot(s*h, W)        
        rval = nnet.sigmoid(
                tensor.add(shW, conv_v_bias))
        return rval
    
    #####################
    # binary spike-and-slab convolutional spike units (h given v)
    def mean_h_given_v(self, v, With_fast):
        alpha = self.get_alpha(With_fast)
        mu = self.get_mu(With_fast)
        W = self.get_filters(With_fast)
        h_bias = self.get_h_bias(With_fast)
        
        vW = self.convdot_T(v, W)
        alpha_vW_mu = vW/alpha + mu 
        rval = nnet.sigmoid(tensor.add(0.5*alpha*(alpha_vW_mu**2),h_bias,-0.5*alpha*(mu**2)))                          
        return rval
        
    #####################
    # binary spike-and-slab convolutional slab units (s given v and h)
    def mean_var_s_given_v_h(self, v, h, With_fast):
        """For efficiency, this method assumes all h variables are 1.
        """
        alpha = self.get_alpha(With_fast)
        mu = self.get_mu(With_fast)
        W = self.get_filters(With_fast)
        
        vW = self.convdot_T(v, W)
        rval = ((vW/alpha)+mu)*h        
        return rval, 1.0 / alpha
   
    #####################

    def gibbs_step_for_s_h(self, s, h, s_rng, return_locals=False, sampling_for_s=True, With_fast=True):
        #positive phase
        # visible variable means
        mean_conv_v = self.mean_conv_v_given_s_h(s, h, With_fast)
        #visible samples
        sample_conv_v = tensor.cast(s_rng.uniform(size=self.out_conv_v_shape) < mean_conv_v, floatX)
        
        #negative phase
        # spike variable means
        mean_h = self.mean_h_given_v(sample_conv_v, With_fast)
        # spike variable samples        
        sample_h = tensor.cast(s_rng.uniform(size=self.hs_shape) < mean_h, floatX)   
        # slab variable means
        meanvar_s = self.mean_var_s_given_v_h(sample_conv_v,sample_h,With_fast)
         # slab variable samples
        mean_s, var_s = meanvar_s 
        if sampling_for_s:
	    random_normal = s_rng.normal(size=self.hs_shape)*tensor.sqrt(var_s)
	    sample_s = random_normal + mean_s
	else:
	    sample_s = mean_s
               
	if return_locals:
            return sample_s, sample_h, locals()
        else:
            return sample_s, sample_h
        
   
    def free_energy_given_s_h(self, s, h, With_fast=False):
        
        alpha = self.get_alpha(With_fast)
        mu = self.get_mu(With_fast)
        W = self.get_filters(With_fast)
        h_bias = self.get_h_bias(With_fast)
        conv_v_bias = self.get_conv_v_bias(With_fast)
        
        out_softplus = 0.5*alpha*(s**2) - alpha*mu*s*h + 0.5*alpha*(mu**2)*h - h_bias*h
        rval = tensor.sum(out_softplus,axis=[1,2,3]) - tensor.sum(nnet.softplus(self.convdot(s*h, W)+conv_v_bias),axis=[1,2,3]) 
        
        assert rval.ndim==1
        return rval
        

    def cd_updates(self, pos_s, pos_h, neg_s, neg_h, stepsizes, other_cost=None):      
        cost=(self.free_energy_given_s_h(pos_s, pos_h, With_fast=False) \
                 - self.free_energy_given_s_h(neg_s, neg_h,With_fast=False)).sum()
        if other_cost:
	    cost = cost + other_cost
	grads = theano.tensor.grad(cost,
		wrt=self.params(),
		consider_constant=[pos_s]+[pos_h]+[neg_s]+[neg_h])
        
        #print len(stepsizes),len(grads+grads)
        assert len(stepsizes)==len(grads+grads)

        if self.l2_conf['unnatural_grad']:
            sgd_updates = unnatural_sgd_updates
        else:
            sgd_updates = pylearn.gd.sgd.sgd_updates
        rval = dict(
                sgd_updates(
                    self.params()+self.params_fast(),
                    grads+grads,
                    stepsizes=stepsizes))
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
        pass
        """
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
        """
    def dump_to_file(self, filename):
        try:
            cPickle.dump(self, open(filename, 'wb'))
        except cPickle.PicklingError:
            pickle.dump(self, open(filename, 'wb'))

class l2_Gibbs(object): # if there's a Sampler interface - this should support it
    @classmethod
    def alloc(cls, brbm, rng):
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self = cls()
        seed=int(rng.randint(2**30))
        self.brbm = brbm
        self.s_particles = sharedX(
            rng.randn(*brbm.hs_shape),
            name='s_particles')
        self.h_particles = sharedX(
            rng.randint(2,size=brbm.hs_shape),
            name='h_particles')    
	#self.particles = sharedX(
        #    numpy.zeros(rbm.v_shape),
        #    name='particles')
	self.s_rng = RandomStreams(seed)
        return self

class l2_Gibbs_for_genrating(object): # if there's a Sampler interface - this should support it
    @classmethod
    def alloc(cls, brbm, rng):
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self = cls()
        seed=int(rng.randint(2**30))
        self.brbm = brbm
        self.v_particles = sharedX(
            rng.randint(2,brbm.out_conv_v_shape),
            name='v_particles')    
	#self.particles = sharedX(
        #    numpy.zeros(rbm.v_shape),
        #    name='particles')
	self.s_rng = RandomStreams(seed)
        return self

class Trainer(object): # updates of this object implement training
    @classmethod
    def alloc(cls, 
            brbm, 
            s_batch,
            h_batch,
            lrdict,
            conf,
            rng=234,
            iteration_value=0,
            ):

        batchsize = brbm.hs_shape[0]
        sampler = l2_Gibbs.alloc(brbm, rng=rng)
	print 'alloc trainer'
        error = 0.0
        return cls(
                brbm=brbm,
                batchsize=batchsize,
                s_batch=s_batch,
                h_batch=h_batch,
                sampler=sampler,
                iteration=sharedX(iteration_value, 'iter'), #float32.....
                learn_rates = [lrdict[p] for p in brbm.params()],
                learn_rates_fast = [lrdict[p_fast] for p_fast in brbm.params_fast()],
                conf=conf,
                annealing_coef=sharedX(1.0, 'annealing_coef'),
                conv_v_means = sharedX(numpy.zeros(brbm.out_conv_v_shape[1:])+0.5,'conv_v_means'),
                conv_v       = sharedX(numpy.zeros(brbm.out_conv_v_shape), 'conv_v'),
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

        conv_v = self.brbm.mean_conv_v_given_s_h(
                self.s_batch, self.h_batch, With_fast=False)
        
        
        new_conv_v_means = 0.1 * conv_v.mean(axis=0) + .9*self.conv_v_means
        ups[self.conv_v_means] = new_conv_v_means
        ups[self.conv_v] = conv_v
        
        #sparsity_cost = 0
        #self.sparsity_cost = sparsity_cost
        # SML updates PCD
        add_updates(
                self.brbm.cd_updates(
                    pos_s=self.s_batch,
                    pos_h=self.h_batch,
                    neg_s=self.sampler.s_particles,
                    neg_h=self.sampler.h_particles,
                    stepsizes=[annealing_coef*lr for lr in self.learn_rates]+[lr_fast for lr_fast in self.learn_rates_fast]))
        if conf['increase_steps_sampling']:
	    steps_sampling = self.iteration.get_value() / 1000 + conf['constant_steps_sampling']
	else:
	    steps_sampling = conf['constant_steps_sampling']
	"""    
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
        """
        
	#print steps_sampling        
        s_tmp_particles = self.sampler.s_particles  
        h_tmp_particles = self.sampler.h_particles
        
        for step in xrange(int(steps_sampling)):
             tmp_particles  = self.brbm.gibbs_step_for_s_h(s_tmp_particles,
                                    h_tmp_particles, self.sampler.s_rng,
                                    sampling_for_s=conf['sampling_for_s'])
             #print tmp_particles                        
             s_tmp_particles, h_tmp_particles = tmp_particles
        new_s_particles = s_tmp_particles 
        new_h_particles = h_tmp_particles
        
        recons_error = 0.0
        ups[self.recons_error] = recons_error
	ups[self.sampler.s_particles] = new_s_particles
        ups[self.sampler.h_particles] = new_h_particles        
        
       
        if conf['alpha_min'] < conf['alpha_max']:
            if conf['alpha_logdomain']:
                ups[self.brbm.alpha] = tensor.clip(
                        ups[self.brbm.alpha],
                        numpy.log(conf['alpha_min']).astype(floatX),
                        numpy.log(conf['alpha_max']).astype(floatX))
                
            else:
                ups[self.brbm.alpha] = tensor.clip(
                        ups[self.brbm.alpha],
                        conf['alpha_min'],
                        conf['alpha_max'])
                        
        weight_decay = numpy.asarray(conf['penalty_for_fast_parameters'], dtype=floatX)
        
        for p_fast in self.brbm.params_fast():
	    new_p_fast = ups[p_fast]
	    new_p_fast = new_p_fast - weight_decay*p_fast
	    ups[p_fast] = new_p_fast
	           
        
        if conf['alpha_min'] < conf['alpha_max']:
            if conf['alpha_logdomain']:
                ups[self.brbm.alpha_fast] = tensor.clip(
                        ups[self.brbm.alpha_fast],
                        numpy.log(conf['alpha_min']).astype(floatX),
                        numpy.log(conf['alpha_max']).astype(floatX))
                
            else:
                ups[self.brbm.alpha_fast] = tensor.clip(
                        ups[self.brbm.alpha_fast],
                        conf['alpha_min'],
                        conf['alpha_max'])       
       
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
        pass
        """
        Image.fromarray(tile_conv_weights(self.sampler.particles.get_value(borrow=True),
            flip=False),'L').save('particles_%s.png'%pattern)
        self.rbm.save_weights_to_grey_files(pattern)
        """
    def print_status(self):
        def print_minmax(msg, x):
            assert numpy.all(numpy.isfinite(x))
            print msg, x.min(), x.max()

        print 'iter:', self.iteration.get_value()
        print_minmax('filters', self.brbm.filters.get_value(borrow=True))
        print_minmax('filters_fast', self.brbm.filters_fast.get_value(borrow=True))
        print_minmax('h_bias', self.brbm.h_bias.get_value(borrow=True))
        print_minmax('h_bias_fast', self.brbm.h_bias_fast.get_value(borrow=True))
        print_minmax('conv_v_bias', self.brbm.conv_v_bias.get_value(borrow=True))
        print_minmax('conv_v_bias_fast', self.brbm.conv_v_bias_fast.get_value(borrow=True))
        print_minmax('mu', self.brbm.mu.get_value(borrow=True))
        print_minmax('mu_fast', self.brbm.mu_fast.get_value(borrow=True))
        if self.conf['alpha_logdomain']:
            print_minmax('alpha',
                    numpy.exp(self.brbm.alpha.get_value(borrow=True)))
            print_minmax('alpha_fast',
                    numpy.exp(self.brbm.alpha_fast.get_value(borrow=True)))
        else:
            print_minmax('alpha', self.brbm.alpha.get_value(borrow=True))
            print_minmax('alpha_fast', self.brbm.alpha_fast.get_value(borrow=True))
        
        print_minmax('s_particles', self.sampler.s_particles.get_value())
        print_minmax('h_particles', self.sampler.h_particles.get_value())
        print_minmax('conv_v_means', self.conv_v_means.get_value())
        print_minmax('conv_v', self.conv_v.get_value())
        
        print (self.conv_v.get_value()).std()
        #print self.conv_h_means.get_value()[0,0:11,0:11]
	#print self.rbm.conv_bias_hs.get_value(borrow=True)[0,0,0:3,0:3]
        #print self.rbm.h_tiled_conv_mask.get_value(borrow=True)[0,32,0:3,0:3]
	#print_minmax('global_h_means', self.global_h_means.get_value())
        print 'lr annealing coef:', self.annealing_coef.get_value()
	#print 'reconstruction error:', self.recons_error.get_value()

def main_sample(layer1_filename, layer2_filename, algo='Gibbs', rng=777888, burn_in=10001, save_interval=5000, n_files=10, sampling_for_v=True):
    rbm = cPickle.load(open(layer1_filename))
    brbm = cPickle.load(open(layer2_filename))
    sampler = l2_Gibbs.alloc(brbm, rng)
    tmp_particles = brbm.gibbs_step_for_s_h(sampler.s_particles,
                               sampler.h_particles, sampler.s_rng,
                               sampling_for_s=brbm.l2_conf['sampling_for_s'])
    s_tmp_particles, h_tmp_particles = tmp_particles 
    n_batchsize, n_maps, n_hs_rows, n_hs_cols = brbm.hs_shape
    icount, fmodules, filters_per_module, hrows, hcols = rbm.out_conv_hs_shape
    assert n_maps==fmodules*filters_per_module
    s_particles_5d = s_tmp_particles.reshape((icount, fmodules, filters_per_module, hrows, hcols))
    h_particles_5d = h_tmp_particles.reshape((icount, fmodules, filters_per_module, hrows, hcols))
    mean_var_samples = rbm.mean_var_v_given_h_s(s_particles_5d, h_particles_5d, True)
    fn = theano.function([], mean_var_samples,
                updates={sampler.s_particles: s_tmp_particles,
                         sampler.h_particles: h_tmp_particles})
                         
    for i in xrange(burn_in):
        print i
        mean_var = fn()
        mean_samples, var_samples = mean_var
        if i % 100 == 0 and i!=0:
	    print 'saving'
	    savename = '%s_DBNsample_burn_%04i.png'%(layer1_filename+layer2_filename,i)
	    Image.fromarray(
                tile_conv_weights(
                    mean_samples[:,:,11:88,11:88],
                    flip=False,scale_each=True),
                'L').save(savename)	    
    """     
    
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

def main0(rval_doc):
    l2_conf = rval_doc['l2_conf']
    rbm = cPickle.load(open(l2_conf['rbm_pkl']))
    sampler = Gibbs.alloc(rbm, rng=33345)    
    batchsize, n_img_channels, \
    n_img_rows, n_img_cols = rbm.v_shape

    batch_idx = tensor.iscalar()
    batch_range = batch_idx*batchsize + numpy.arange(batchsize)
    
    batch_x = Brodatz_op(batch_range,
  	                 l2_conf['dataset'],   # download from http://www.ux.uis.no/~tranden/brodatz.html
  	                 patch_shape=rbm.v_shape[1:], 
  	                 noise_concelling=0., 
  	                 seed=3322, 
  	                 batchdata_size=rbm.v_shape[0],
                         rescale=1.0,
                         rescale_size=[rbm.conf['data_rescale'],]
  	                )	           
    brbm = bRBM.alloc(
            l2_conf,
            hs_shape=(
                 rbm.out_conv_hs_shape[0],
                 rbm.out_conv_hs_shape[1]*rbm.out_conv_hs_shape[2],
                 rbm.out_conv_hs_shape[3],
                 rbm.out_conv_hs_shape[4]
                 ),
            filters_shape=(                 
                l2_conf['n_filters'],
                rbm.out_conv_hs_shape[1]*rbm.out_conv_hs_shape[2],
                l2_conf['filters_size'],
                l2_conf['filters_size']
                ),            #fmodules(stride) x filters_per_modules x fcolors(channels) x frows x fcols
            filters_irange=l2_conf['filters_irange'],
            rbm=rbm,
            )
    brbm.save_weights_to_grey_files('layer2_iter_0000')

    base_lr = l2_conf['base_lr_per_example']/batchsize
    conv_lr_coef = l2_conf['conv_lr_coef']
    h_mean = rbm.mean_convhs_h_given_v(batch_x, With_fast=False)
    s_mean_var = rbm.mean_var_convhs_s_given_v(batch_x, With_fast=False)
    s_mean, s_var = s_mean_var
    batchsize, fmodules, filters_per_module, hrows, hcols = rbm.out_conv_hs_shape
        
    if l2_conf['fast_weights']:
        trainer = Trainer.alloc(
            brbm,
            s_batch=s_mean.reshape((batchsize, fmodules*filters_per_module, hrows, hcols)),
            h_batch=h_mean.reshape((batchsize, fmodules*filters_per_module, hrows, hcols)),
            lrdict={
                brbm.filters: sharedX(conv_lr_coef*base_lr, 'filters_lr'),
                brbm.conv_v_bias: sharedX(base_lr, 'conv_v_bias_lr'),
                brbm.h_bias: sharedX(base_lr, 'h_bias_lr'),
                brbm.mu: sharedX(base_lr, 'mu_lr'),
                brbm.alpha: sharedX(base_lr, 'alpha_lr'),
                brbm.filters_fast: sharedX(conv_lr_coef*base_lr, 'filters_fast_lr'),
                brbm.conv_v_bias_fast: sharedX(base_lr, 'conv_v_bias_fast_lr'),
                brbm.h_bias_fast: sharedX(base_lr, 'h_bias_fast_lr'),
                brbm.mu_fast: sharedX(base_lr, 'conv_mu_fast_lr'),
                brbm.alpha_fast: sharedX(base_lr, 'conv_alpha_fast_lr')
                },
            conf = l2_conf,
            )            
    else:
        trainer = Trainer.alloc(
            brbm,
            s_batch=s_mean.reshape((batchsize, fmodules*filters_per_module, hrows, hcols)),
            h_batch=h_mean.reshape((batchsize, fmodules*filters_per_module, hrows, hcols)),
            lrdict={
                brbm.filters: sharedX(conv_lr_coef*base_lr, 'filters_lr'),
                brbm.conv_v_bias: sharedX(base_lr, 'conv_v_bias_lr'),
                brbm.h_bias: sharedX(base_lr, 'h_bias_lr'),
                brbm.mu: sharedX(base_lr, 'mu_lr'),
                brbm.alpha: sharedX(base_lr, 'alpha_lr'),
                brbm.filters_fast: sharedX(0.0, 'filters_fast_lr'),
                brbm.conv_v_bias_fast: sharedX(0.0, 'conv_v_bias_fast_lr'),
                brbm.h_bias_fast: sharedX(0.0, 'h_bias_fast_lr'),
                brbm.mu_fast: sharedX(0.0, 'conv_mu_fast_lr'),
                brbm.alpha_fast: sharedX(0.0, 'conv_alpha_fast_lr')
                },
            conf = l2_conf,
            )  

  
    print 'start building function'
    training_updates = trainer.updates() #
    train_fn = theano.function(inputs=[batch_idx],
            outputs=[],
	    #mode='FAST_COMPILE',
            #mode='DEBUG_MODE',
	    updates=training_updates	    
	    )  #

    print 'training the second layer...'
    
    iter = 0
    while trainer.annealing_coef.get_value()>=0: #
        dummy = train_fn(iter) #
        if iter % 10 == 0:
	    trainer.print_status()
	if iter % 1000 == 0:
            brbm.dump_to_file(os.path.join(_temp_data_path_,'brbm_%06i.pkl'%iter))
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
       l2_conf=dict(
            dataset='../../Brodatz/D6.gif',
            rbm_pkl='./rbm_040000.pkl', 
            #chain_reset_prob=0.0,#reset for approximately every 1000 iterations #we need scan for the burn in loop
            #chain_reset_iterations=100
            #chain_reset_burn_in=0,
            unnatural_grad=False,
            alpha_logdomain=False,
            alpha0=10.,           
            alpha_min=1.,
            alpha_max=1000.,           
            mu0 = 1.0,
            train_iters=40000,
            base_lr_per_example=0.00001,
            conv_lr_coef=1.0,
            n_filters=64,            
	    filters_size=2,
            filters_irange=.001,            
            #sparsity_weight_conv=0,#numpy.float32(500),
            #sparsity_weight_global=0.,
            particles_min=-1000.,
            particles_max=1000.,
            constant_steps_sampling = 1,         
            increase_steps_sampling = False,
            sampling_for_s=True,
            penalty_for_fast_parameters = 0.1,
            fast_weights = False
            )))
    

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        sys.exit(main_train())
    if sys.argv[1] == 'sampling':
	sys.exit(main_sample(sys.argv[2],sys.argv[3]))    
    