import cv2 as cv
import numpy as np
from decimal import Decimal
import os

path ='D:\Year 3\Plant_Images'
files = os.listdir(path)

variable=1
successful=0
unsuccessful=0
asp_ratio_min=1.99
asp_ratio_max=1.20

for filename in files:
    if filename.endswith('.JPG'):
        image_name = os.path.join(path,filename)
        print(image_name)
        image=cv.imread(image_name)
        
        #Converting RGB to HSV values so only green color is segmented from the image
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        #Defining the range of green values

        #lower Mask(90-100)
        lower_green=np.array([90,50,50])
        upper_green=np.array([100,255,255])
        mask0 = cv.inRange(image_hsv, lower_green, upper_green)
        
        #Upper Mask(170,180)
        lower_green=np.array([170,50,50])
        uppper_green=np.array([180,255,255])
        mask1=cv.inRange(image_hsv,lower_green, upper_green)
        
        #Joining the two masks
        mask = mask0+mask1
        
        #Output after masking
        output_hsv=image_hsv.copy()
        output_hsv[np.where(mask=0)]=0
        
        
        
        
        