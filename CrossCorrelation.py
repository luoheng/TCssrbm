import numpy
    
class CrossCorrelation(object):
    
    def __init__(self, test_image, samples, window_size, seed=98987, n_patches_of_samples=0):
        self.test_image = test_image
        self.samples = samples
        #assert self.test_image.dtype == self.samples.dtype
        self.window_size = window_size
        self.rng = numpy.random.RandomState(seed)
        
        n_samples,channels_samples,rows_samples,cols_samples = self.samples.shape
        channels_test,rows_test,cols_test = self.test_image.shape
        assert channels_test == channels_samples
	assert rows_test >= rows_samples
	assert cols_test >= cols_samples
	assert rows_samples >= window_size
	assert cols_samples >= window_size
	if rows_samples>window_size or cols_samples>window_size:
	    assert n_patches_of_samples>0
	    self.patches = numpy.zeros((n_samples,n_patches_of_samples,channels_samples,window_size,window_size),self.samples.dtype)
	    for samples_index in xrange(n_samples):
                offsets_row = self.rng.randint(rows_samples-window_size+1, size=n_patches_of_samples)
                offsets_col = self.rng.randint(cols_samples-window_size+1, size=n_patches_of_samples)
                for n, (r,c) in enumerate(zip(offsets_row, offsets_col)):
	            temp_patch= self.samples[samples_index,:,r:r+window_size,c:c+window_size]
	            temp_patch = temp_patch/numpy.sqrt((temp_patch**2).sum())
	            self.patches[samples_index,n,:,:,:] = temp_patch
	else:
	    self.patches = numpy.zeros((n_samples,1,channels_samples,rows_samples,cols_samples))
	    for samples_index in xrange(n_samples):
		self.patches[samples_index,0,] = samples[samples_index,]/numpy.sqrt((samples[samples_index,]**2).sum())
        
        
       
    def NCC(self):
        channels_test,rows_test,cols_test = self.test_image.shape
        n_samples,n_patches,channels_patches,rows_patches,cols_patches = self.patches.shape        
	
	rc_test_img = numpy.zeros((rows_test-rows_patches+1,cols_test-cols_patches+1,channels_test,rows_patches,cols_patches))
	for row_index in xrange(rows_test-rows_patches+1):
	    for col_index in xrange(cols_test-cols_patches+1):
		temp_patch = self.test_image[:,row_index:row_index+rows_patches,col_index:col_index+cols_patches]
		rc_test_img[row_index,row_index,] = temp_patch/numpy.sqrt((temp_patch**2).sum())
	
	value_NCC = numpy.zeros((n_samples,n_patches,rows_test-rows_patches+1,cols_test-cols_patches+1))
	for samples_index in xrange(n_samples):
	    for n_patches_index in xrange(n_patches):
	        rc_patch = self.patches[samples_index,n_patches_index,]
		for row_index in xrange(rows_test-rows_patches+1):
		    for col_index in xrange(cols_test-cols_patches+1):
			temp_patch = rc_test_img[row_index,row_index,]		    
			value_NCC[samples_index,n_patches_index,row_index,col_index] = numpy.dot(
			  rc_patch.reshape(1,channels_patches*rows_patches*cols_patches),
			  temp_patch.reshape(1,channels_patches*rows_patches*cols_patches).T)
			  
        self.value_NCC = value_NCC
        return value_NCC
        
    def TSS(self):
         try:
	     value_NCC = self.value_NCC
	 except:
             value_NCC = self.NCC()
         
         return numpy.amax(numpy.amax(numpy.amax(value_NCC,1),1),1)
		      
		      
	       