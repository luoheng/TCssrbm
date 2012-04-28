"""Convenience base classes to help with writing Dataset ops
To randomly generate data, we use 
try:
    x = self.x_
except:
    x = self.fn(*self.fn_args)
 
"""

__docformat__  = "restructuredtext_en"
import numpy
import theano

class Dataset(theano.Op):
    """ 
    The basic dataset interface is an expression that maps an integer to a dataset element.

    There is also a minibatch option, in which the expression maps an array of integers to a
    list or array of dataset elements.
    """
    def __init__(self, single_type, batch_type):
        self.single_type = single_type
        self.batch_type = batch_type

    def make_node(self, idx):
        _idx = theano.tensor.as_tensor_variable(idx)
        if not _idx.dtype.startswith('int'):
            raise TypeError()
        if _idx.ndim == 0: # one example at a time
            otype = self.single_type
        elif _idx.ndim == 1: #many examples at a time
            otype = self.batch_type
        else:
            raise TypeError(idx)
        return theano.Apply(self, [_idx], [otype()])

    def __eq__(self, other):
        return type(self) == type(other) \
                and self.single_type == other.single_type \
                and self.batch_type == other.batch_type

    def __hash__(self):
        return hash(type(self)) ^ hash(self.single_type) ^ hash(self.batch_type)

    def __str__(self):
        return "%s{%s,%s}" % (self.__class__.__name__, self.single_type, self.batch_type)

    def grad(self, inputs, g_outputs):
        return [None for i in inputs]


class TensorDataset(Dataset):
    """A convenient base class for Datasets whose elements all have the same TensorType.
    """
    def __init__(self, dtype, single_broadcastable, single_shape=None, batch_size=None):
        single_broadcastable = tuple(single_broadcastable)
        self.single_shape = single_shape
        self.batch_size = batch_size
        single_type = theano.tensor.Tensor(
                broadcastable=single_broadcastable,
                dtype=dtype)
        batch_type = theano.tensor.Tensor(
                broadcastable=(False,)+single_type.broadcastable,
                dtype=dtype)
        super(TensorDataset, self).__init__(single_type, batch_type)
    def __eq__(self, other):
        return (super(TensorDataset, self).__eq__(other)
                and self.single_shape == other.single_shape
                and self.batch_size == other.batch_size)
    def __hash__(self):
        return (super(TensorDataset, self).__hash__()
                ^ hash(self.single_shape)
                ^ hash(self.batch_size))

class TensorFnDataset(TensorDataset):
    """A good base class for TensorDatasets that are backed by indexed objects.
    E.g. numpy ndarrays and memmaps.

    This Op looks up the dataset by a function call, rather than by storing it
    as a member variable.  This is done to make the graph serializable without
    having to save the dataset itself, which is typically large.

    This Op is picklable if (and only if) the function that accesses the dataset
    can be serialized.
    """
    def __init__(self, dtype, bcast, fn, single_shape=None, batch_size=None):
        """
        :type fn: callable or (callable, args) tuple [MUST BE PICKLABLE!]
        :param fn: function that returns the dataset as a ndarray-like object.

        :type bcast: tuple of bool
        :param bcast: the broadcastable flag for the return value if this op is
            indexed by a scalar (the one example case)  A (False,) will be
            pre-pended to this pattern when the Op is indexed by a vector.
        """
        super(TensorFnDataset, self).__init__(dtype, bcast, single_shape, batch_size)
        try:
            self.fn, self.fn_args = fn
        except:
            self.fn, self.fn_args = fn, ()
    def __getstate__(self):
        rval = dict(self.__dict__)
        if 'x_' in rval:
            del rval['x_']
        return rval

    def __eq__(self, other):
        return super(TensorFnDataset, self).__eq__(other) and self.fn == other.fn \
                and self.fn_args == other.fn_args

    def __hash__(self):
        return (super(TensorFnDataset, self).__hash__()
                ^ hash(self.fn)
                ^ hash(self.fn_args))

    def __str__(self):
        try:
            return "%s{%s,%s}" % (self.__class__.__name__, self.fn.__name__, self.fn_args)
        except:
            return "%s{%s}" % (self.__class__.__name__, self.fn, self.fn_args)

    def perform(self, node, (idx,), (z,)):
        try:
            x = self.x_
        except:
            x = self.fn(*self.fn_args)
        if idx.ndim == 0:
            z[0] = numpy.asarray(x[int(idx)]) # asarray is important for memmaps
        else:
            z[0] = numpy.asarray(x[idx]) # asarray is important for memmaps
