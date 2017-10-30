# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import pyflow
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

def createFlow(video, impath1, impath2):
    im1 = np.array(Image.open(os.path.join(base,video, impath1)))
    im2 = np.array(Image.open(os.path.join(base,video, impath2)))

    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()

    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    import cv2
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(savebase, video, impath1), rgb)


def main():
    futures = set()
    with ProcessPoolExecutor() as executor:
        for video in list:

            if os.path.isdir(os.path.join(savebase, video)) != True:
                os.mkdir(os.path.join(savebase, video))

            frames = os.listdir(os.path.join(base, video))
            frames.sort()

            for i in range(len(frames) - 1):
                future = executor.submit(createFlow, video, frames[i], frames[i+1])
                futures.add(future)
    try:
        for future in as_completed(futures):
            err = future.exception()
            if err is not None:
                raise err
    except KeyboardInterrupt:
        print("stopped by hand")


if __name__ == '__main__':

    base = '/home/link/rgbFrames'
    savebase = '/home/link/flowFrames'

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    list = os.listdir(base)
    if os.path.isdir(savebase) != True:
        os.mkdir(savebase)

    main()
