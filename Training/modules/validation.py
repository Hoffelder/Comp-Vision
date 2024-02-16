from sklearn.metrics import f1_score,precision_score,recall_score ,accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import random
import os 
import json 
from model import Model, EfficientDet
from tqdm import tqdm

class SklearnValidate:

     def eval(self,model,X_test,X_train,y_train,y_test):

        self.cv   = StratifiedKFold(n_splits=10, random_state=41, shuffle=True)        
        cv_result = cross_val_score(model,X_train,y_train, cv=self.cv, n_jobs=-1)

        y_hat_test   = model.predict(X_test)
        y_hat_train  = model.predict(X_train)

        metrics = {
            "10-fold accuracy":      cv_result.mean().round(3),
            "10-fold deviation":     cv_result.std().round(3),

            "train accuracy_score":  accuracy_score(  y_train, y_hat_train ).round(3),
            "train f1_score":        f1_score(        y_train, y_hat_train ).round(3),
            "train precision_score": precision_score( y_train, y_hat_train ).round(3),
            "train recall_score":    recall_score(    y_train, y_hat_train ).round(3),

            "test accuracy_score":   accuracy_score(  y_test, y_hat_test ).round(3),
            "test f1_score":         f1_score(        y_test, y_hat_test ).round(3),
            "test precision_score":  precision_score( y_test, y_hat_test ).round(3),
            "test recall_score":     recall_score(    y_test, y_hat_test ).round(3)
        }

        return metrics 

class TfLiteValidate:
    def eval(self,model,test_data):
        return model.evaluate(test_data)


class Validate(TfLiteValidate):
    def __init__(self) -> None:
        self.iou = IoU()
        
    
    def eval(self,model,test_data,path, name):

        efficientDet = EfficientDet(path+"/"+name+"/model.tflite")
        IoUs, dets, tresholds = self.iou.get_metrics(efficientDet)

        return model.evaluate(test_data),IoUs, dets, tresholds


class IoU:
    def __init__(self):
        self.df            = pd.read_csv("whole.csv")
        self.validation_df = self.df[self.df.iloc[:,0] == "VALIDATION"]
        self.paths         = self.validation_df.iloc[:,1].unique().tolist()

    def get_centers(self,boxes):
        centers = []
        for bb in boxes:

            x0 = float(bb[0])
            y0 = float(bb[1])
            x1 = float(bb[2])           
            y1 = float(bb[3])
            
            xc = round(abs((x0 + abs(x1-x0)/2)),4) 
            yc = round(abs((y0 + abs(y1-y0)/2)),4)

            centers.append((xc,yc))

        return np.array(centers)

    def reshape(self,img):
        detection_frame        = cv2.resize(img, (320,320), interpolation = cv2.INTER_AREA)
        detection_frame        = np.expand_dims(detection_frame,axis=0).astype(np.uint8)
        return detection_frame

    def bb_intersection_over_union(self,boxA, boxB):
        
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def get_IoUs(self,detection_boxes,scores,label_bboxes,threshold = 0.5):

        center_det    = self.get_centers(detection_boxes[scores>threshold])
        center_labels = self.get_centers(label_bboxes)
        distances     = np.zeros((center_labels.shape[0],center_det.shape[0]))

        for i, det_c in enumerate(center_det):
            for j,label_c in enumerate(center_labels):
                distances[j,i] = round((((label_c[0]-det_c[0])**2+(label_c[1]-det_c[1])**2)**0.5),3)

        matches = np.argwhere(distances<0.02)

        IoUs = []
        for m in matches:
            label_index,d_index = m
            IoU = self.bb_intersection_over_union(detection_boxes[d_index],label_bboxes[label_index])
            IoUs.append(IoU)
        
        mean_IoU = sum(IoUs)/len(IoUs) if len(IoUs)> 0 else 0
        detection_accuracy = max(1 - abs(len(detection_boxes[scores>threshold])-len(label_bboxes))/len(label_bboxes),0)
        
        return mean_IoU,detection_accuracy

    def get_labels(self,df,path):

        labels = df[df.iloc[:,1] == path].iloc[:,3:]
        y      = (np.array(labels)[:,[1,3,5,7]])
        x      = (np.array(labels)[:,[0,2,4,6]])
        x_min  = x[:,0].reshape(-1,1) 
        y_min  = y[:,0].reshape(-1,1)
        x_max  = x[:,1].reshape(-1,1)
        y_max  = y[:,2].reshape(-1,1)
        bboxes = np.concatenate([x_min, y_min, x_max, y_max],axis = 1)
        scores = np.array([1]*len(bboxes))
        return bboxes,scores

    def get_metrics(self,model):

        all_IoUs  = []
        all_dets  = []
        tresholds = []

        for path in tqdm(self.paths):
            img                 = cv2.imread(path)[:,:,::-1]
            detection_frame     = self.reshape(img)
            label_bboxes,scores = self.get_labels(self.df,path)
            detection_boxes, classes, scores = model.predict(detection_frame)
            IoUs = []
            dets = []
            for t in np.linspace(0,1,50):
                IoU,det = self.get_IoUs(detection_boxes,scores,label_bboxes,threshold = t)
                IoUs.append(IoU)
                dets.append(det)
            
            all_IoUs.append(np.array(IoUs))
            all_dets.append(np.array(dets))
        
        tresholds = np.linspace(0,1,50).tolist()
        mean_iou  = np.array(all_IoUs).mean(axis = 0)
        mean_det  = np.array(all_dets).mean(axis = 0)

        print("********************************")
        print(len(tresholds))
        print(mean_det.shape)
        print(mean_iou.shape)
        print("********************************")
        return mean_iou, mean_det, tresholds



   