import theano
import numpy
from PIL import Image
from protocol_ import TensorFnDataset

floatX=theano.config.floatX

def Brodatz_op(s_idx, filename, patch_shape=(1,98,98), noise_concelling=0.0, seed=3322, batchdata_size=64, rescale=1.0, rescale_size=(2,)):
    """Return symbolic Brodatz_images[s_idx]

    If s_idx is a scalar, the return value is a tensor3 of shape 1,98,98.
    If s_idx is a vector of len N, the return value
    is a tensor4 of shape N,1,98,98.
    """
    assert len(filename)==len(rescale_size)
    ob = Brodatz(filename, patch_shape, noise_concelling, seed, batchdata_size, rescale, rescale_size)
    fn = ob.extract_random_patches
    op = TensorFnDataset(floatX,
            bcast=(False, False, False),
            fn=fn,
            single_shape=(1,98,98))
    return op(s_idx%(batchdata_size*len(filename)))
    
class Brodatz(object):
    
    def __init__(self, filename, patch_shape, noise_concelling, seed, batchdata_size, rescale, rescale_size):        
        self.patch_shape = patch_shape
        self.filename = filename
        self.ncc = noise_concelling
        self.rng = numpy.random.RandomState(seed)
        self.batchdata_size = batchdata_size
        self.training_img = []
        self.test_img = []        
        f_index = 0
        for f_name in self.filename:
	    image = Image.open(f_name)
	    image_rows, image_cols = image.size
	    image = image.resize((int(image_rows/rescale_size[f_index]),int(image_cols/rescale_size[f_index])),Image.BICUBIC)
	    img_array = numpy.asarray(image, dtype=floatX)
            training_img = numpy.zeros((int(image_rows/(2*rescale_size[f_index])),int(image_cols/rescale_size[f_index])),dtype=floatX)
            test_img = numpy.zeros((int(image_rows/(2*rescale_size[f_index])),int(image_cols/rescale_size[f_index])))
            training_img = img_array[0:int(image_rows/(2*rescale_size[f_index])),:]
            test_img = img_array[int(image_rows/(2*rescale_size[f_index])):,:]
               
	    patch_channels, patch_rows, patch_cols = patch_shape        
	    assert patch_rows < int(image_rows/(2*rescale_size[f_index]))  
	    assert patch_cols < int(image_cols/(rescale_size[f_index])) 
	    assert patch_channels == 1        
            self.training_img += [(training_img - training_img.mean())/(rescale*training_img.std()+self.ncc)]
            self.test_img += [test_img]        
            print 'the std of the training data %s is:%f' %(f_name, self.training_img[f_index].std()) 
            f_index += 1
             
    #@staticmethod   
    def extract_random_patches(self):
        N = self.batchdata_size
        _, patch_rows, patch_cols = self.patch_shape
        rval = numpy.zeros((N*len(self.training_img),1,patch_rows,patch_cols), dtype=self.training_img[0].dtype) 
        #print rval.shape        
        for img_index in xrange(len(self.training_img)):
	    img = self.training_img[img_index]
	    img_rows, img_cols = img.shape	  
	    offsets_row = self.rng.randint(img_rows-patch_rows+1, size=N)
            offsets_col = self.rng.randint(img_cols-patch_cols+1, size=N)
            for n, (r,c) in enumerate(zip(offsets_row, offsets_col)):
	        rval[img_index*N+n,0,:,:] = img[r:r+patch_rows,c:c+patch_cols]	    
	    #temp_img = rval_temp
	    #temp_img = numpy.asarray(255*(temp_img - temp_img.min()) / (temp_img.max() - temp_img.min() + 1e-6),
            #            dtype='uint8')
            #Image.fromarray(temp_img[0,0],'L').save('ptches_inner_%s.png'%img_index)	    
        return rval
      
