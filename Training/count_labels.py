import json
import cv2  
import numpy as np
import os

all_files = os.listdir("god_10_14out")


total = 0

for file in all_files:

    f = open('god_10_14out/'+str(file))
    data = json.load(f)
    f.close()

    h,w = 200,600

    points = []
    for shapes in data["shapes"]:
        x0,y0 = np.array(shapes["points"])[0]
        x1,y1 = np.array(shapes["points"])[1]
        points.append([x0/w, y0/h,  x1/w, y0/h, x1/w, y1/h, x0/w, y1/h])
        
    pos = np.array(points)
    for bb in pos:
        y = (bb[[1,3,5,7]]*h).astype(int)
        x = (bb[[0,2,4,6]]*w).astype(int)

    print("total labels:", len(pos))

    total += len(pos)

print("TOTAL:", total, "\ntotal imags:",len(all_files),"\nlabels/img:",total/len(all_files), "\nlabels/h", (total/40),"\nlabels/min",(total/40/60),"\ns/label",60/(total/40/60))