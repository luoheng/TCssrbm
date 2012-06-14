import theano
import numpy
from PIL import Image
from protocol_ import TensorFnDataset

floatX=theano.config.floatX

def Brodatz_op(s_idx, filename, patch_shape=(1,98,98), noise_concelling=100, seed=3322, batchdata_size=64, rescale=1.0, validation=False):
    """Return symbolic Brodatz_images[s_idx]

    If s_idx is a scalar, the return value is a tensor3 of shape 1,98,98.
    If s_idx is a vector of len N, the return value
    is a tensor4 of shape N,1,98,98.
    """
    
    ob = Brodatz(filename, patch_shape, noise_concelling, seed, batchdata_size, rescale, validation)
    fn = ob.extract_random_patches
    op = TensorFnDataset(floatX,
            bcast=(False, False, False),
            fn=fn,
            single_shape=(1,98,98))
    return op(s_idx%batchdata_size)
    
class Brodatz(object):
    
    def __init__(self, filename, patch_shape, noise_concelling, seed, batchdata_size, rescale, validation=False):        
        self.patch_shape = patch_shape
        self.filename = filename
        self.ncc = noise_concelling
        self.rng = numpy.random.RandomState(seed)
        self.batchdata_size = batchdata_size
        image = Image.open(filename)
        image_rows, image_cols = image.size
        image = image.resize((int(image_rows/2),int(image_cols/2)), Image.BICUBIC)
        new_image_rows, new_image_cols = image.size
        img_array = numpy.asarray(image, dtype=floatX)
        
        if validation:
            # The model is in validation mode, the training set is going to be the 2/3 of
            # the top half of the image and the testing set is going to be the remaining third
            train_validation_limit = int(new_image_cols*2/3)
            self.training_img = numpy.zeros(img_array[0:int(new_image_rows/2),:train_validation_limit].shape,dtype=floatX)
            self.test_img =     numpy.zeros(img_array[0:int(new_image_rows/2),train_validation_limit:].shape)
            self.training_img = img_array[0:int(new_image_rows/2),:train_validation_limit]
            self.test_img =     img_array[0:int(new_image_rows/2),train_validation_limit:]
        else:
            # The model is in test mode, the training set is going to be the whole
            # top half of the image and the testing set is going to be the bottom half            
            self.training_img = numpy.zeros((int(new_image_rows/2),new_image_cols),dtype=floatX)
            self.test_img = numpy.zeros((int(new_image_rows/2),new_image_cols))
            self.training_img = img_array[0:int(new_image_rows/2),:]
            self.test_img = img_array[int(new_image_rows/2):,:]
        
        print "BrodatzOp : using a validation set : " + str(validation)
        print "BrodatzOp : the training image size is : " + str(self.training_img.shape)
        print "BrodatzOp : the test image size is : " + str(self.test_img.shape)
                
        patch_channels, patch_rows, patch_cols = patch_shape        
        assert patch_rows < int(new_image_rows/2)  
        assert patch_cols < int(new_image_cols) 
        assert patch_channels == 1
        
        self.training_img = self.training_img - self.training_img.mean()
        self.training_img = self.training_img/(rescale*self.training_img.std()+self.ncc)
        
        #self.test_img = self.test_img - self.test_img.mean()
        #self.test_img = self.test_img/(rescale*self.test_img.std()+self.ncc)
        print 'the std of the training data is:%f' %self.training_img.std() 
        #print 'the std of the test data is:%f' %self.test_img.std() 
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
      
