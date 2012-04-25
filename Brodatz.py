import theano
import numpy
from PIL import Image
from protocol import TensorFnDataset

floatX=theano.config.floatX

def Brodatz_op(s_idx, filename, patch_shape=(1,98,98), noise_concelling=100, seed=3322, batchdata_size=64):
    """Return symbolic Brodatz_images[s_idx]

    If s_idx is a scalar, the return value is a tensor3 of shape 1,98,98.
    If s_idx is a vector of len N, the return value
    is a tensor4 of shape N,1,98,98.
    """
    
    ob = Brodatz(filename, patch_shape, noise_concelling, seed, batchdata_size)
    fn = ob.extract_random_patches
    op = TensorFnDataset(floatX,
            bcast=(False, False, False),
            fn=fn,
            single_shape=(1,98,98))
    return op(s_idx)
    
class Brodatz(object):
    
    def __init__(self, filename, patch_shape, noise_concelling, seed, batchdata_size):        
        self.patch_shape = patch_shape
        self.filename = filename
        self.ncc = noise_concelling
        self.rng = numpy.random.RandomState(seed)
        self.batchdata_size = batchdata_size
        image = Image.open(filename)
        image_rows, image_cols = image.size
        image = image.resize((int(image_rows/2),int(image_cols/2)))
        img_array = numpy.asarray(image, dtype=floatX)
        self.training_img = numpy.zeros((int(image_rows/4),int(image_cols/2)),dtype=floatX)
        #self.test_img = numpy.zeros((int(image_rows/4),int(image_cols/2)))
        self.training_img = img_array[0:int(image_rows/4),:]
        #self.test_img = img_array[int(image_rows/4):,:]
        
        patch_channels, patch_rows, patch_cols = patch_shape        
        assert patch_rows < int(image_rows/4)  
        assert patch_cols < int(image_cols/2) 
        assert patch_channels == 1
        
        self.training_img = self.training_img - self.training_img.mean()
        self.training_img = self.training_img/(self.training_img.std()+self.ncc)
    #@staticmethod   
    def extract_random_patches(self):
        N = self.batchdata_size
        _, patch_rows, patch_cols = self.patch_shape
        img_rows, img_cols = self.training_img.shape
	rval = numpy.zeros((N,1,patch_rows,patch_cols), dtype=self.training_img.dtype)
               
        offsets_row = self.rng.randint(img_rows-patch_rows+1, size=N)
        offsets_col = self.rng.randint(img_cols-patch_cols+1, size=N)

        for n, (r,c) in enumerate(zip(offsets_row, offsets_col)):
	     rval[n,0,:,:] = self.training_img[r:r+patch_rows,c:c+patch_cols]
        return rval
      