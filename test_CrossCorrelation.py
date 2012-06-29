import numpy
import theano
from CrossCorrelation import CrossCorrelation

floatX=theano.config.floatX
arrayX = lambda X : numpy.asarray(X, dtype=floatX)

def rand(shp, dtype=floatX):
    return numpy.random.rand(*shp).astype(dtype)

test_image_shp = (1,4,4)
n_sample=2
n_patches_of_samples=1
samples_shp = (n_sample,n_patches_of_samples,2,2)
window_size=2
seed=0
test_image = arrayX(numpy.ones(test_image_shp))
samples = rand(samples_shp)
print 'test_image'
print test_image
print 'samples'
print samples
print samples.shape

CC = CrossCorrelation(test_image,samples,window_size,n_patches_of_samples=4)
print CC.patches
print CC.patches.shape
NCC_value = CC.NCC()
print NCC_value
for n in xrange(n_sample):
    print ((samples[n,0:window_size,0:window_size]/numpy.sqrt((samples[n,0,0:window_size,0:window_size]**2).sum()))*0.5).sum()
        
#assert (samples[0,0,:,:]/numpy.sqrt((samples[0,0,:,:]**2).sum())*0.5).sum() == NCC_value[0,0]
