import json
import cv2  
import numpy as np
import os


f = open('IMG_4003.json')
data = json.load(f)
f.close()


img = cv2.imread("Solida/"+data["imagePath"])

h,w,c = img.shape

print("w,h,c",w,h,c)
print(data["imagePath"])

points = []
for shapes in data["shapes"]:
    x0,y0 = np.array(shapes["points"])[0]
    x1,y1 = np.array(shapes["points"])[1]
    points.append([x0/w, y0/h,  x1/w, y0/h, x1/w, y1/h, x0/w, y1/h])
    
pos = np.array(points)
for bb in pos:
    print(bb)
    y = (bb[[1,3,5,7]]*h).astype(int)
    x = (bb[[0,2,4,6]]*w).astype(int)

    print("LABELME TO TF:",["Solida/IMG_4003.json"]+["log"]+list(bb) )

    cv2.rectangle(img,(x[0],y[0]),(x[1],y[2]),(0,255,0),2)

img = img.astype(np.uint8)  

#img = cv2.resize(img, (int(w/8),int(h/8)), interpolation = cv2.INTER_AREA)

#cv2.imshow("frame",img)
#cv2.waitKey(0)



import labelme 
import base64

data       = labelme.LabelFile.load_image_file("Solida/IMG_4003.JPG")
image_data = base64.b64encode(data).decode('utf-8')

print(image_data)







["Solida/IMG_4003.json"]+["log"]+list(bb) 
  
import labelme 
import base64

data       = labelme.LabelFile.load_image_file("Solida/IMG_4001.JPG")
image_data = base64.b64encode(data).decode('utf-8')

print(image_data)

x0,y0 = np.array(shapes["points"])[0]
x1,y1 = np.array(shapes["points"])[1]

x0,y0,x1,y1 = np.array(points)[0][[0,1,2,5]]

print(x0,y0,x1,y1)
#points.append([x0/w, y0/h,  x1/w, y0/h, x1/w, y1/h, x0/w, y1/h])



new_point = {
            "label": "solida",
            "points": [
                [
                    int(x0*w), 
                    int(y0*h)
                ],
                [
                    int(x1*w), 
                    int(y1*h)
                ]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
            }



{
    "version": "5.0.1",
    "flags": {},
    "shapes": [
        new_point,
        new_point
    ],
    "imagePath":   "IMG_4003.JPG",
    "imageData":   image_data,
    "imageHeight": 3024,
    "imageWidth":  4032
}