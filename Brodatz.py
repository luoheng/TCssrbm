import theano
import numpy
from PIL import Image
from protocol_ import TensorFnDataset

floatX=theano.config.floatX

def Brodatz_op(s_idx, 
               filename, 
               patch_shape=(1,98,98), 
               noise_concelling=0.0, 
               seed=3322, 
               batchdata_size=64, 
               rescale=1.0, 
               rescale_size=(2,),
               validation=False,
               test_data=False
               ):
    """Return symbolic Brodatz_images[s_idx]

    If s_idx is a scalar, the return value is a tensor3 of shape 1,98,98.
    If s_idx is a vector of len N, the return value
    is a tensor4 of shape N,1,98,98.
    """
    assert len(filename)==len(rescale_size)
    ob = Brodatz(filename, 
                 patch_shape, 
                 noise_concelling, 
                 seed, 
                 batchdata_size, 
                 rescale, 
                 rescale_size, 
                 validation)
    fn = ob.extract_random_patches
    op = TensorFnDataset(floatX,
            bcast=(False, False, False),
            fn=fn,
            single_shape=(1,98,98))
    return op(s_idx%(batchdata_size*len(filename)))
    
class Brodatz(object):
    
    def __init__(self, filename, 
                 patch_shape, noise_concelling, 
                 seed, batchdata_size, rescale, 
                 rescale_size,
                 validation=False,
                 test_data=False):        
        self.patch_shape = patch_shape
        self.filename = filename
        self.ncc = noise_concelling
        self.rng = numpy.random.RandomState(seed)
        self.batchdata_size = batchdata_size
        self.training_img = []
        self.validation = validation
        self.test_data = test_data
        self.test_img = []        
        f_index = 0
        patch_channels, patch_rows, patch_cols = patch_shape 
        for f_name in self.filename:
	    image = Image.open(f_name)
	    image_rows, image_cols = image.size
	    image = image.resize((int(image_rows/rescale_size[f_index]),int(image_cols/rescale_size[f_index])),Image.BICUBIC)
	    img_array = numpy.asarray(image, dtype=floatX)           
                        
            if validation:
		# The model is in validation mode, the training set is going to be the 2/3 of
		# the top half of the image and the testing set is going to be the remaining third
		train_rows = int(image_rows/(2*rescale_size[f_index]))
		train_cols = int(image_cols/rescale_size[f_index]*2/3)
		training_img = numpy.zeros((train_rows,train_cols),dtype=floatX)
		test_rows = train_rows
		test_cols = int(image_cols/rescale_size[f_index]*1/3)
		test_img = numpy.zeros((test_rows,test_cols))
		training_img = img_array[0:train_rows,0:train_cols]
		test_img = img_array[train_rows:,train_cols:]		
		assert patch_rows < train_rows  
		assert patch_cols < train_cols 
	    else:
		# The model is in test mode, the training set is going to be the whole
		# top half of the image and the testing set is going to be the bottom half     
		training_img = numpy.zeros((int(image_rows/(2*rescale_size[f_index])),int(image_cols/rescale_size[f_index])),dtype=floatX)
		test_img = numpy.zeros((int(image_rows/(2*rescale_size[f_index])),int(image_cols/rescale_size[f_index])))
		training_img = img_array[0:int(image_rows/(2*rescale_size[f_index])),:]
		test_img = img_array[int(image_rows/(2*rescale_size[f_index])):,:]	
		assert patch_rows < int(image_rows/(2*rescale_size[f_index]))  
		assert patch_cols < int(image_cols/(rescale_size[f_index])) 
        
	    print "BrodatzOp : using a validation set : " + str(validation)
	    print "BrodatzOp : the training image size is : " + str(training_img.shape)
	    print "BrodatzOp : the test image size is : " + str(test_img.shape)           
	    
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
	    if self.test_data:
	        img = self.test_img[img_index]	        
	    else:    
	        img = self.training_img[img_index]
	    img_rows, img_cols = img.shape	  
	    offsets_row = self.rng.randint(img_rows-patch_rows+1, size=N)
            offsets_col = self.rng.randint(img_cols-patch_cols+1, size=N)
            for n, (r,c) in enumerate(zip(offsets_row, offsets_col)):
	        if self.test_data:
		    temp = img[r:r+patch_rows,c:c+patch_cols]
		    temp = (temp - temp.mean())/temp.std()    #we normalized pathes not the whole test image
		    rval[img_index*N+n,0,:,:] = temp
		else:
		    rval[img_index*N+n,0,:,:] = img[r:r+patch_rows,c:c+patch_cols]	    
	    #temp_img = rval_temp
	    #temp_img = numpy.asarray(255*(temp_img - temp_img.min()) / (temp_img.max() - temp_img.min() + 1e-6),
            #            dtype='uint8')
            #Image.fromarray(temp_img[0,0],'L').save('ptches_inner_%s.png'%img_index)	    
        return rval
      
