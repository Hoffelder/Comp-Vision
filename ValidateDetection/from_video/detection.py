import cv2
import numpy as np
import tensorflow as tf
from run_tf_light import Model
import time


#exp2_.tflite
model = Model("models/1-0.tflite" )

test_files = {
          "0":'videos/pilha_pixlo_2.MOV',
          "1":'videos/sombra.mp4',
          "2":'videos/umbara1.mp4',
          "3":'videos/umbara2.mp4',
          "4":'videos/umbara3.mp4'
}

confidence = 0.4
img_size   = 384
cap   = cv2.VideoCapture(test_files["3"]) 

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

def resize(frame,scale_percent = 50):
     # percent of original size
    h,w,c = frame.shape
    height = 960
    p = h/height
    width = w/p

    print(height,width)
    #width         = int(frame.shape[1] * scale_percent / 100)
    #height        = int(frame.shape[0] * scale_percent / 100)
    dim           = (int(width), int(height))
    frame         = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    print(frame.shape)
    return frame

def ChangeScaleFloat(y):
    return y * (newMax-newMin) + newMin



def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)
    


    start_y = boxes[:, 0]
    start_x = boxes[:, 1]
    end_y   = boxes[:, 2]
    end_x   = boxes[:, 3]

    # coordinates of bounding boxes
    #start_x = boxes[:, 0]
    #start_y = boxes[:, 1]
    #end_x   = boxes[:, 2]
    #end_y   = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    #print(order)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        #print(ratio)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  
  if ret == True:
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = resize(frame)

    h,w,c = frame.shape
    h0 = int((h-w)/2)
    hf = int(h0+w)
    
    # Display the resulting frame
    #cv2.imshow('Frame',frame)

    crop = frame[h0:hf,:,:]

    if cv2.waitKey(25) & 0xFF == ord('q'):
      time.sleep(10)
      break
    
    crop2 = frame[h0:hf,:,:]
    
    detection_frame        = cv2.resize(crop2, (img_size,img_size), interpolation = cv2.INTER_AREA)
    detection_frame        = np.expand_dims(detection_frame,axis=0).astype(np.uint8)
    boxes, classes, scores = model.predict(detection_frame)

    m = 0.1
    frame = cv2.rectangle(frame, (int(w*m),int(h0+h*m)), (int(w*(1-m) ),int(hf*(1-m))), (255,255,255), 3)

    picked_boxes, picked_score = nms(boxes,scores,2)
    conflist = []
    for bb,c,s in zip(picked_boxes, picked_score,picked_score):
        conflist.append(s)
        #print(len(conflist),len(set(conflist)))
        if s>confidence:
            
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
            r  = int((abs(y1-y0)/2 + abs(x1-x0)/2)/2)    
            Xc = xc
            Yc = yc


            #frame = cv2.rotate(frame, cv2.ROTATE_180)
            #frame = cv2.putText(frame, str(round(s,2)), (xc,yc), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                  (0,0,255), 2, cv2.LINE_AA, True)
            #frame = cv2.rotate(frame, cv2.ROTATE_180)

            if xc>int(w*m) and xc<int(w*(1-m)) and yc<int(hf*(1-m)) and yc > int(h0+h*m): 
              frame = cv2.circle(frame, (xc,yc), r , (0,0,255), 2)  
            else:
              frame = cv2.circle(frame, (xc,yc), r , (255,0,0), 2)  

            


            ''' 
            #crop = cv2.circle(crop, (xc,yc), r , (255,255,255), 2)
            #cv2.imshow('Crop',crop)

            x_0 = int(bb[0])
            y_0 = int(bb[1])
            x_1 = int(bb[2])            
            y_1 = int(bb[3])

            padding = abs(h-w)
            newMin  = padding/2/h 
            newMax  = padding/2 + w/h 
            y_0 = y_0*(newMax-newMin) + newMin
            y_1 = y_1*(newMax-newMin) + newMin
            y_0 *=h 
            y_1 *=h 
            x_0 *=w  
            x_1 *=w

            y_0 = int(h - y_0)
            y_1 = int(h - y_1)

            print( (x_0,y_0), (x_1,y_1))

            frame = cv2.rectangle(frame, (x_0,y_0), (x_1,y_1), (0,0,255), 3) 

            #xc = int(x0 + abs(x1-x0)/2) 
            #yc = int(y0 + abs(y1-y0)/2)
            #r  = int(abs(y1-y0)/2)       
            # 
            ''' 
            
            

            
    
    cv2.imshow('Frame',frame)

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
