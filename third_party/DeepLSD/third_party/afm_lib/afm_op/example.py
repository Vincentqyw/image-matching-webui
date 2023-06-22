import torch
import afmop

import os
import os.path as osp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from tqdm import tqdm
from lib.afm.gpu_afm import afm_transform_gpu as afm_transform

data_root = '../data/wireframe_raw'
output_root = '../data/wireframe'


with open(osp.join(data_root,'train.txt')) as handle:
    train_lst = [f.rstrip('.jpg\n') for f in handle.readlines()]

with open(osp.join(data_root,'test.txt')) as handle:
    test_lst = [f.rstrip('.jpg\n') for f in handle.readlines()]


def load_datum(filename, height = 0, width = 0, mirror = 0):
    with open(osp.join(data_root,'pointlines',filename+'.pkl'),'rb') as handle:
        d = pickle.load(handle, encoding='latin1')
        h, w = d['img'].shape[:2]
        points = d['points']
        lines = d['lines']
        lsgs = np.array([[points[i][0], points[i][1], points[j][0], points[j][1]] for i, j in lines],
                        dtype=np.float32)
        image = d['img']
                
        return image, lsgs

BATCH_SIZE = 16
data = [load_datum(f) for f in train_lst[:BATCH_SIZE]]

lines = np.concatenate([data[i][1] for i in range(BATCH_SIZE)])
start = np.array([data[i][1].shape[0] for i in range(BATCH_SIZE)])
end   = np.cumsum(start)
start = end-start
shape_info = np.array([[start[i], end[i], data[i][0].shape[0],data[i][0].shape[1]] for i in range(BATCH_SIZE) ])

lines = torch.Tensor(lines).cuda()
shape_info = torch.IntTensor(shape_info).cuda()
# shape_info = np.array([[0,data[i][1], data[i][0].shape[0], data[i][0].shape[1]] for i in range(4), data[i][0].shape[0], data[i][0].shape[1]]],dtype=np.float32)
import time
start = time.time()
for i in range(3000):
    afmap, aflabel = afmop.afm(lines, shape_info, 375,500)
print((time.time()-start))

for i in range(BATCH_SIZE):
    xx, yy = np.meshgrid(range(500),range(375))
    im_tensor = torch.Tensor(data[i][0].transpose([2,0,1])).unsqueeze(0)

    im_tensor = torch.nn.functional.interpolate(im_tensor,size=[375,500],mode='bilinear',align_corners=False)
    afx = afmap[i][0].data.cpu().numpy() + xx
    afy = afmap[i][1].data.cpu().numpy() + yy    
    image = im_tensor.data.cpu().numpy()[0].transpose([1,2,0])/255.0
    plt.imshow(image)
    plt.plot(afx,afy,'r.',markersize=0.5)
    plt.show()
