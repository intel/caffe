import h5py
import numpy as np
import skvideo.datasets
import skvideo.io

videodata = skvideo.io.vread('1.avi')
count = 0
data = videodata
data=data.transpose(3,0,1,2) # To chanelxdepthxhxw
data=data[None,:,:,:]
print(videodata)

with h5py.File('data.h5','w') as f:
   f['data'] = data
   f['label'] = np.ones((1,1), dtype=np.float32)
