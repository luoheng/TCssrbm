import numpy
from PIL import Image


def most_square_shape(N):
    """rectangle (height, width) with area N that is closest to sqaure
    """
    for i in xrange(int(numpy.sqrt(N)),0, -1):
        if 0 == N % i:
            return (i, N/i)


def tile_conv_weights(w,flip=False, scale_each=False):
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

   