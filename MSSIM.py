import numpy
    
class MSSIM(object):
    
    def __init__(self, test_samples, samples, window_size, seed=98987):
                         
        self.test_samples = test_samples
        self.samples = samples
        self.window_size = window_size
        self.rng = numpy.random.RandomState(seed)
        
        self.single_test_sample = len(test_samples.shape) == 3
        
        if self.single_test_sample:
            # Only one test sample
            n_samples,channels_samples,rows_samples,cols_samples = self.samples.shape
            channels_test,rows_test,cols_test = self.test_samples.shape
            
            # Ensure coherence between the shapes of the inputs
            assert channels_test == channels_samples
            assert rows_test == rows_samples
            assert cols_test == cols_samples       
            
        else:
            # One test sample by sample
            n_samples,channels_samples,rows_samples,cols_samples = self.samples.shape
            n_tests, channels_test,rows_test,cols_test = self.test_samples.shape
            
            # Ensure coherence between the shapes of the inputs
            assert channels_test == channels_samples
            assert rows_test == rows_samples
            assert cols_test == cols_samples       
            assert n_samples == n_tests
        
         
    def MSSIM_old(self):
        # Return the mean of the SSIM scores for all the samples
        if self.single_test_sample:
            sumSSIM = sum([self.SSIM(s, self.test_samples)
                           for s in self.samples])
            return (1.0 / len(self.samples)) * sumSSIM            
        else:
            sumSSIM = sum([self.SSIM(self.samples[i], self.test_samples[i])
                           for i in range(len(self.samples))])
            return (1.0 / len(self.samples)) * sumSSIM
            
    def MSSIM(self):
        # Return the mean and the std_dev of the SSIM scores for all the samples
        #import pdb
        #pdb.set_trace()
        
        SSIMs = None
        if self.single_test_sample:
            SSIMs = [self.MSSIM_one_sample(s, self.test_samples) for s in self.samples]          
        else:
            SSIMs = [self.MSSIM_one_sample(self.samples[i], self.test_samples[i]) for i in range(len(self.samples))]
            
        return (numpy.mean(SSIMs), numpy.std(SSIMs))
            
    def MSSIM_one_sample(self, sample, test_sample):
        results = numpy.zeros((sample.shape[-3], sample.shape[-2] + 1 - self.window_size, sample.shape[-1] + 1 - self.window_size))
        for color in range(results.shape[0]):
            for row in range(results.shape[1]):
                for col in range(results.shape[2]):
                    results[color, row, col] = self.SSIM(sample[:,row:row+self.window_size, col:col+self.window_size],
                                                test_sample[:,row:row+self.window_size, col:col+self.window_size])
        return numpy.mean(results)
                
        
    def SSIM(self, sample, test_sample):

               
        meanSample = numpy.mean(sample)
        meanTest = numpy.mean(test_sample)
        
        stdSample = numpy.std(sample)
        stdTest = numpy.std(test_sample)
        
        
        covariance = numpy.cov(sample.flatten(), test_sample.flatten())[0,1]        
        covariance = numpy.nan_to_num(covariance)
        

        
        C1 = (255 * 0.01) ** 2 # Constant to avoid instability when 
                               # meanSample**2 + meanTest**2 approaches 0
        C2 = (255 * 0.03) ** 2 # Constant to avoid instability when 
                               # stdSample**2 + stdTest**2 approaches 0 
        
        numerator = (2 * meanSample * meanTest + C1) * (2 * covariance + C2)
        denominator = (meanSample ** 2 + meanTest ** 2 + C1) * \
                      (stdSample ** 2 + stdTest ** 2 + C2)
        
        return numerator / denominator


