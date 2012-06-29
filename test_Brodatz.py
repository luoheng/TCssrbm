import theano
from Brodatz import Brodatz_op, Brodatz
import numpy
from PIL import Image
from TCssrbm import tile_conv_weights

s_idx=theano.tensor.lscalar()
batch_range =s_idx*128 + numpy.arange(128)
n_examples = 16
n_img_channels = 1
n_img_rows = 98
n_img_cols = 98

 

batch_x = Brodatz_op(batch_range,
  	             ['../Brodatz/D21.gif',
  	             '../Brodatz/D6.gif',
  	             '../Brodatz/D53.gif',
  	             '../Brodatz/D77.gif',
  	             '../Brodatz/D4.gif',
  	             '../Brodatz/D16.gif',
  	             '../Brodatz/D68.gif',
  	             '../Brodatz/D103.gif'],   # download from http://www.ux.uis.no/~tranden/brodatz.html
  	             patch_shape=(n_img_channels,
  	                          n_img_rows,
  	                          n_img_cols), 
  	             noise_concelling=0., 
  	             seed=3322, 
  	             batchdata_size=n_examples,
                     rescale=1.0,
                     new_shapes = [[320,320],
                                   [480,480],
                                   [320,320],
                                   [480,480],
                                   [480,480],
                                   [640,640],
                                   [320,320],
                                   [320,320]
		       ],
		     validation=False,    
                     test_data=False   
  	            )	
fn=theano.function([s_idx],batch_x)


B_texture = Brodatz(['../Brodatz/D6.gif',
  	             '../Brodatz/D21.gif',
  	             '../Brodatz/D53.gif',
  	             '../Brodatz/D77.gif',
  	             '../Brodatz/D4.gif',
  	             '../Brodatz/D16.gif',
  	             '../Brodatz/D68.gif',
  	             '../Brodatz/D103.gif'],   # download from http://www.ux.uis.no/~tranden/brodatz.html
  	             patch_shape=(n_img_channels,
  	                          n_img_rows,
  	                          n_img_cols), 
  	             noise_concelling=0., 
  	             seed=3322, 
  	             batchdata_size=n_examples,
                     rescale=1.0,
                     new_shapes = [[320,320],
                                   [480,480],
                                   [320,320],
                                   [480,480],
                                   [480,480],
                                   [640,640],
                                   [320,320],
                                   [320,320]
                     validation=False,    
                     test_data=False              
		   ])
"""
for ii in xrange(8):
    shp = B_texture.test_img[ii].shape
    #img = numpy.zeros((1,)+shp)
    temp_img = numpy.asarray(B_texture.test_img[ii], dtype='uint8')
    #img[0,] = temp_img
    Image.fromarray(temp_img,'L').save('test_img_%s.png'%ii)  
    
    shp = B_texture.training_img[ii].shape
    #img = numpy.zeros((1,)+shp)
    temp_img = numpy.asarray(255*(B_texture.training_img[ii] - B_texture.training_img[ii].min()) / (B_texture.training_img[ii].max() - B_texture.training_img[ii].min() + 1e-6),
                        dtype='uint8')
    #img[0,] = temp_img
    Image.fromarray(temp_img[0],'L').save('training_img_%s.png'%ii)
"""
for n in xrange(10):
    img_1=fn(n)
    #img_2=fn(n+1)
    #assert img_1.any() == img_2.any()
    Image.fromarray(tile_conv_weights(img_1,
            flip=False),'L').save('patches_%s.png'%n)
