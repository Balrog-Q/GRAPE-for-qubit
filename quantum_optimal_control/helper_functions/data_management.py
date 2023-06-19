import numpy as np
import h5py
import json
import datetime
import sys

class H5File(h5py.File):
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        self.flush()

    def add(self, key, data):
        data = np.array(data)
        data = data[np.newaxis]
        dt = h5py.special_dtype(vlen=str)
        
        try:
            self.create_dataset(key,
                                shape=data.shape,
                                maxshape=tuple([None] * len(data.shape)),
                                dtype=dt if (data.dtype == '<U1' or data.dtype == '<U8') else str(data.dtype))
        except ValueError:
            del self[key]
            self.create_dataset(key,
                                shape=data.shape,
                                maxshape=tuple([None] * len(data.shape)),
                                dtype=dt if (data.dtype == '<U1' or data.dtype == '<U8') else str(data.dtype))

        self[key][...] = data
    
    def append(self, key, data, forceInit=False):
        data = np.array(data)
        data = data[np.newaxis]
        dt = h5py.special_dtype(vlen=str)
                            
        try:
            self.create_dataset(key,
                                shape=tuple([1] + list(data.shape)),
                                maxshape=tuple([None] * (len(data.shape) + 1)),
                                dtype=dt if (data.dtype == '<U1' or data.dtype == '<U8') else str(data.dtype))
        except ValueError:
            if forceInit == True:
                del self[key]
                self.create_dataset(key,
                                    shape=tuple([1] + list(data.shape)),
                                    maxshape=tuple([None] * (len(data.shape) + 1)),
                                    dtype=dt if (data.dtype == '<U1' or data.dtype == '<U8') else str(data.dtype))
            
            dataset = self[key]
            shape = list(dataset.shape)
            shape[0] = shape[0] + 1
            dataset.resize(shape)
        
        dataset = self[key]

        try:
            dataset[-1,:] = data
        except TypeError:
            dataset[-1] = data