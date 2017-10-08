# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import os
#parser = argparse.ArgumentParser(
#    description='Demo for python wrapper of Coarse2Fine Optical Flow')
#parser.add_argument(
#    '-viz', dest='viz', action='store_true',
#    help='Visualize (i.e. save) output of flow.')
#args = parser.parse_args()

def createFlow(base, savebase):
    list = os.listdir(base)
    if os.path.isdir(savebase) != True:
        os.mkdir(savebase)
    
    for item in list:
        videos = os.listdir(os.path.join(base, item))
        for video in videos:
            frames = os.listdir(os.path.join(base, item, video))
            frames.sort(key= lambda x:int(x[:-4]))
            if os.path.isdir(os.path.join(savebase, item, video)) != True:
                 os.makedirs(os.path.join(savebase, item, video))
            
            if len(frames) > 1:
                im1 = np.array(Image.open(os.path.join(base, item, video, frames[0])))
                im1 = im1.astype(float) / 255.
            for i in range(1, len(frames)):
                im2 = np.array(Image.open(os.path.join(base, item, video, frames[i])))
                im2 = im2.astype(float) / 255.

                # Flow Options:
                alpha = 0.012
                ratio = 0.75
                minWidth = 20
                nOuterFPIterations = 7
                nInnerFPIterations = 1
                nSORIterations = 30
                colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

                s = time.time()
                u, v, im2W = pyflow.coarse2fine_flow(
                    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                    nSORIterations, colType)
                e = time.time()
                print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
                    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
                flow = np.concatenate((u[..., None], v[..., None]), axis=2)
                #np.save('examples/outFlow.npy', flow)


                import cv2
                hsv = np.zeros(im1.shape, dtype=np.uint8)
                hsv[:, :, 0] = 255
                hsv[:, :, 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(os.path.join(savebase, item, video, frames[i-1]), rgb)
                #cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)
                
                im1 = im2

createFlow('/home/link/frames','/home/link/flow_images')



