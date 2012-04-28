import theano
from Brodatz import Brodatz_op
import numpy
from PIL import Image
from TCssrbm import tile_conv_weights

s_idx=theano.tensor.lscalar()
batch_range =s_idx*64 + numpy.arange(64)
batchdat=Brodatz_op(batch_range,'../Brodatz/D6.gif')
fn=theano.function([s_idx],batchdat)
for n in xrange(10):
    img_1=fn(n)
    img_2=fn(n+1)
    #assert img_1.any() == img_2.any()
    Image.fromarray(tile_conv_weights(img_1,
            flip=False),'L').save('D6patches_%s.png'%n)

