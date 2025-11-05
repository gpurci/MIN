#!/usr/bin/python

import numpy as np
import cv2

def reshape(image, height=None, width=None):
   h, w = image.shape[:2]
   size = np.array(image.shape[:2], dtype=np.float32)
   if (height != None):
      zoom = height / h
   elif (width != None):
      zoom = width / w
   else:
      zoom = 600 / h
   #print('zoom {}'.format(zoom))
   new_size = size * zoom
   new_size = new_size.astype(np.int32)

   #print('size {}'.format(image.shape))
   #print('size {}, new size {}, zoom {}'.format(size, new_size, zoom))
   new_h, new_w = new_size
   show_image = cv2.resize(image, (new_w, new_h), 
            interpolation = cv2.INTER_AREA)
   return show_image

