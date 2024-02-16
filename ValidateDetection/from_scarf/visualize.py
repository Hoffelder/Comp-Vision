import cv2
import numpy as np
import pandas as pd
import time


def resize(frame,scale_percent = 50):
     # percent of original size
    width         = int(frame.shape[1] * scale_percent / 100)
    height        = int(frame.shape[0] * scale_percent / 100)
    dim           = (width, height)
    frame         = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame

def ChangeScaleFloat(y):
    return y * (newMax-newMin) + newMin


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


"""
[

    x0,y0 ------
    |          |
    |          |
    |          |
    --------x1,y1
  
'Pixel X1':         proporção x0 na imagem original
'Pixel Y1':         proporção y0 na imagem original
'Pixel X2':         proporção x1 na imagem original      
'Pixel Y2':         proporção y1 na imagem original  
'Scale X1':         pixel x0 na imagem original    
'Scale Y1':         pixel y0 na imagem original 
'Scale X2':         pixel x1 na imagem original  
'Scale Y2':         pixel y1 na imagem original  
'PreRescaled Y1':   proporção y0 na imagem cropada  
'PreRescaled Y2':   proporção y1 na imagem cropada
]

"""

# Read until video is completed
for i in range(6):
  # Capture frame-by-frame
  file_path_crop     = "BB/ImageCropped"  + str(i) + ".jpg" 
  file_path_original = "BB/ImageOriginal" + str(i) + ".jpg" 
  file_path_resized  = "BB/ImageResized"  + str(i) + ".jpg"

  frame              = cv2.imread(file_path_original)
  crop               = cv2.imread(file_path_crop)
  resized            = cv2.imread(file_path_resized)

  print("************************************",resized.shape)

  df                 = pd.read_csv("BB/Vertices-"+str(i)+".csv",sep = ";")
  
  
  #cv2.imshow('original',frame)
  #cv2.imshow('crop',crop)
  #cv2.imshow('resized',resized)

  #print(df)
  #print(df.columns)
  #exit()
  #print(np.array(df))

  h,w,c = frame.shape
  h0 = int((h-w)/2)
  hf = int(h0+w)
  
  # Display the resulting frame
  #cv2.imshow('Frame',frame)

  crop_ = frame[h0:hf,:,:]

  #cv2.imshow('crop',crop)
  #cv2.imshow('crop_____',crop_)
  #cv2.waitKey(10000)

  boxes = np.array(df[['Pixel X1', 'Pixel Y1', 'Pixel X2', 'Pixel Y2']])

  print(boxes)

  for bb in boxes:
     y0,x0,y1,x1,  xc,yc,r = fit_boxes(frame,bb,h0 = 0 )
     frame = cv2.circle(frame, (xc,yc), r , (0,0,255), 4)  
  
  boxes = np.array(df[['Scale X1','Scale Y1','Scale X2','Scale Y2']])
  for bb in boxes:
    x0, y0, x1, y1 = bb
    xc    = int(x0 + abs(x1-x0)/2) 
    yc    = int(y0 + abs(y1-y0)/2)
    r     = int(abs(y1-y0)/2) 
    frame = cv2.circle(frame, (xc,yc), r , (255,255,255), 2)    




  boxes = np.array(df[['Pixel X1','PreRescaled Y1','Pixel X2','PreRescaled Y2']])
  for bb in boxes:
    x0, y0, x1, y1 = bb
    y0,x0,y1,x1,  xc,yc,r = fit_boxes(crop,bb,h0 = 0 )
    crop = cv2.circle(crop, (xc,yc), r , (255,255,255), 2)    

  'PreRescaled Y1'
  'PreRescaled Y2'

  'Pixel X1'
  'Pixel Y1'


      #y0,x0,y1,x1,  xc,yc,r = fit_boxes(resized,bb,h0 = h0)
    #frame = cv2.circle(resized, (xc,yc), r , (255,255,255), 2)  
    #y0,x0,y1,x1,  xc,yc,r = fit_boxes(resized,bb,h0 = h0)
    #frame = cv2.circle(frame, (xc,yc), r , (255,255,255), 2)  
    #cv2.imshow('crop',crop)
    #cv2.imshow('resized',resized)  


  #y0,x0,y1,x1,  xc,yc,r = fit_boxes(resized,bb,h0 = h0)
  #frame = cv2.circle(resized, (xc,yc), r , (255,255,255), 2)


  cv2.imshow('original',frame)
  cv2.imshow('crop',crop)
  cv2.waitKey(5000)
    
      
cv2.destroyAllWindows()
