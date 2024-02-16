import cv2
import numpy as np
import tensorflow as tf
from model import Model
import time
import pandas as pd 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

model = Model()

def fit_boxes(crop,bb,h0 = 0):

  #x0 = bb[0]
  #y0 = bb[1]
  #x1 = bb[2]
  #y1 = bb[3]   

  h,w,c = crop.shape

  print("h,w,c",h,w,c)

  x0    = int(bb[0]*w)
  y0    = int(bb[1]*h)#+h0

  x1    = int(bb[2]*w) 

  y1    = int(bb[3]*h)#+h0           
   
  xc    = int(x0 + abs(x1-x0)/2) 
  yc    = int(y0 + abs(y1-y0)/2)
  r     = int(abs(y1-y0)/2)   

  print(y0,x0,y1,x1,  xc,yc,r) 

  return y0,x0,y1,x1,  xc,yc,r

def resize(frame,scale_percent = 50):
     # percent of original size
    width         = int(frame.shape[1] * scale_percent / 100)
    height        = int(frame.shape[0] * scale_percent / 100)
    dim           = (width, height)
    frame         = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame

def ChangeScaleFloat(y):
    return y * (newMax-newMin) + newMin


for i in range(6):
  # Capture frame-by-frame
  file_path_crop     = "BB/ImageCropped"  + str(i) + ".jpg" 
  file_path_original = "BB/ImageOriginal" + str(i) + ".jpg" 
  file_path_resized  = "BB/ImageResized"  + str(i) + ".jpg"



  frame              = cv2.imread(file_path_original)
  crop               = cv2.imread(file_path_crop)
  resized            = cv2.imread(file_path_resized)

  scarf_frame = frame.copy()

  h,w,c = frame.shape
  h0 = int((h-w)/2)
  hf = int(h0+w)
  
  crop = frame[h0:hf,:,:]

  
  crop2 = frame[h0:hf,:,:]
  
  detection_frame  = cv2.resize(crop2, (320,320), interpolation = cv2.INTER_AREA)
  detection_frame  = np.expand_dims(detection_frame,axis=0).astype(np.uint8)
  boxes, classes, scores = model.predict(detection_frame)

  for bb,c,s in zip(boxes, classes, scores):
      if s>0.5:
          
          '''
          x0,y0 ------
          |          |
          |          |
          |          |
          --------x1,y1
          '''
          
          #crop = cv2.rectangle(crop, (x0,y0), (x1,y1), (0,0,255), 3)

          h,w,c = crop.shape
          y0 = int(bb[0]*h)+h0
          x0 = int(bb[1]*w)
          y1 = int(bb[2]*h)+h0           
          x1 = int(bb[3]*w)  

          xc = int(x0 + abs(x1-x0)/2) 
          yc = int(y0 + abs(y1-y0)/2)
          r  = int(abs(y1-y0)/2)    
          Xc = xc
          Yc = yc
          
          frame = cv2.circle(frame, (xc,yc), r , (255,0,0), 2)    


  df = pd.read_csv("BB/Vertices-"+str(i)+".csv",sep = ";")
  
  boxes = np.array(df[['Pixel X1', 'Pixel Y1', 'Pixel X2', 'Pixel Y2']])

  print(boxes)

  for bb in boxes:
     y0,x0,y1,x1,  xc,yc,r = fit_boxes(frame,bb,h0 = 0 )
     scarf_frame = cv2.circle(frame, (xc,yc), r , (0,0,255), 2)  
       
          
      

          
  
  cv2.imshow('Model',frame)
  #cv2.imshow('Scarf',scarf_frame)
  
  cv2.waitKey(5000)



# Closes all the frames
cv2.destroyAllWindows()
