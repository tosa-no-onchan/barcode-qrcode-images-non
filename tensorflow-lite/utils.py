# -*- coding: utf-8 -*-
'''
 copy from
 guichristmann/edge-tpu-tiny-yolo
 utils.py
 
 update by nishi 2021.2.25

'''
import numpy as np
import os
import cv2


'''
  letterbox_image(imagex, size):
    size : width , height
 update by nishi 2021.2.25
'''
def letterbox_image(imagex, size,func=0):
    # add by nishi
    image=cv2.cvtColor(imagex,cv2.COLOR_BGR2RGB)
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[0:2][::-1]
    w, h = size   # tensor input width , height
    #print('w,h:',w,h)
    #print('iw,ih:',iw,ih)
    if func==0:
        scale = min(w/iw, h/ih)
        # test by nishi
        #print(">>scale = %f"%(scale))
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.zeros((size[1], size[0], 3), np.uint8)
        new_image.fill(128)
        dx = (w-nw)//2
        dy = (h-nh)//2
        new_image[dy:dy+nh, dx:dx+nw,:] = image
        v_size=(int(w/scale),int(h/scale))      # virtual image size (w,h)

    else:
        #scale = max(w/iw, h/ih)
        scale_w = w/iw
        scale_h = h/ih
        # test by nishi
        #print(">>scale = %f"%(scale))
        nw = int(iw*scale_w)
        nh = int(ih*scale_h)
        new_image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
        #print('new_image.shape:',new_image.shape)
        v_size=(iw,ih)      # virtual image size (w,h)

    return new_image,v_size
