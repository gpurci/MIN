#!/usr/bin/python

import cv2
from reshape import *
import matplotlib.pyplot as plt

def image_show(image, window_name="Test", ):
   show_image = reshape(image, height=900)
   # Using cv2.imshow() method 
   # Displaying the image 
   cv2.imshow(window_name, show_image) 
    
def image_show_wait(image, window_name="Test", ):
   image_show(image, window_name)
   # waits for user to press any key 
   # (this is necessary to avoid Python kernel form crashing) 
   key = cv2.waitKey(0)
   print("Key {}".format(key))

   # closing all open windows 
   cv2.destroyAllWindows() 

def imshow(image, name=None, filename=None):
   fig = plt.imshow(image)
   fig.set_cmap('brg')
   fig.axes.get_xaxis().set_visible(False)
   fig.axes.get_yaxis().set_visible(False)
   if (name is not None):
      pass
      #fig.title(name)
   if (filename is not None):
      plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
   plt.show()
