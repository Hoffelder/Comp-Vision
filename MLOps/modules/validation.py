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
from modules.models import Models, EfficientDet
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

        IoUs, dets, tresholds = self.iou.get_metrics(efficientDet,metrics_path = path+"/"+name+"/"+name+".csv")

        return model.evaluate(test_data),IoUs, dets, tresholds


class IoU:

    def bb_intersection_over_union(self, boxA, boxB):
        
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

    def get_IoUs(self, detection_boxes,scores,label_bboxes,confidence_threshold = 0.5,IoU_threshold = 0.5,max_detections = 100):
        
        distances     = np.zeros((len(detection_boxes),len(label_bboxes)))
        for i, det_c in enumerate(detection_boxes):
            if scores[i]>confidence_threshold:
                #print(scores[i])
                for j,label_c in enumerate(label_bboxes):
                    distances[i,j] = self.bb_intersection_over_union(label_bboxes[j],detection_boxes[i])
        #print(i)
        matches = np.argwhere(distances>IoU_threshold)
        #print(distances)
        IoUs = []
        hits = np.array(range(len(label_bboxes)))*0
        
        for m in matches:
            detection_index,label_index = m
            #print(distances[detection_index,label_index])
            IoUs.append(distances[detection_index,label_index])
            hits[label_index]+=1
        
        mean_IoU = sum(IoUs)/len(IoUs) if len(IoUs)> 0 else 0
        detection_accuracy = (hits==1).sum()/(min(len(hits),max_detections))
        
        #max(1 - abs(len(detection_boxes[scores>threshold])-len(label_bboxes))/len(label_bboxes),0)
        #print(hits,detection_accuracy,confidence_threshold,mean_IoU)
        
        return mean_IoU,detection_accuracy


    def proportion_to_pixel(self, img, labels):
        #print("-------------------------------- SHAPE --------------------------------",np.array(labels).shape)
        
        label = np.array(labels)
        h,w,c = img.shape
        if np.array(labels).shape[1]==8:
            label[:,[1,3,5,7]] = label[:,[1,3,5,7]]*h
            label[:,[0,2,4,6]] = label[:,[0,2,4,6]]*w
            label = label.astype(int)
            label = label[:,[1,0,5,2]]

        else:
            label[:,[1,3]] = label[:,[1,3]]*h
            label[:,[0,2]] = label[:,[0,2]]*w
            label = label.astype(int)
            
        return label

    def reshape(self, img, size = 384):
        detection_frame        = cv2.resize(img, (size,size), interpolation = cv2.INTER_AREA)
        detection_frame        = np.expand_dims(detection_frame,axis=0).astype(np.uint8)
        return detection_frame

    def draw_detection(self, boxes, classes, scores,mock, threshold=0.5, color = 'r'):
        h,w,c = mock.shape
        
        
        options = {'red':  (255,0,0),
                'blue': (0,0,255),
                'green':(0,255,0),
                'black':(0,0,0),
                'white':(255,255,255)}

        size  =  2
        
        for bb,c,s in zip(boxes, classes, scores):
            #print(bb)
            if s>threshold:

                '''
                x0,y0 ------
                |          |
                |          |
                |          |
                --------x1,y1
                '''
                
                y0 = int(bb[0])
                x0 = int(bb[1])
                y1 = int(bb[2])           
                x1 = int(bb[3])  
                

                xc = abs(int(x0 + abs(x1-x0)/2)) 
                yc = abs(int(y0 + abs(y1-y0)/2))
                r  = max(int(abs(y1-y0)/2),int(abs(x1-x0)/2)) 
                Xc = xc
                Yc = yc
                
                #mock = mock.astype(np.uint8)
                mock = cv2.circle(mock, (xc,yc), r , options[color], 4)
        return mock

    def remove_bb_out_of_image(self, bb, h, w):
        mask = np.ones(bb.shape[0])
        for i in range(4):
            mask *= (bb[:,:]>0)[:,i]
        bb = bb[mask.astype(bool),:]

        mask = np.ones(bb.shape[0])
        for i in [0,2]:
            mask *= (bb[:,:]<w)[:,i]
        bb = bb[mask.astype(bool),:]

        mask = np.ones(bb.shape[0])
        for i in [1,3]:
            mask *= (bb[:,:]<h)[:,i]
        bb = bb[mask.astype(bool),:]

    def get_labels(self, test_dataset, path):
        return test_dataset[test_dataset.iloc[:,1] == path].iloc[:,3:]
    
    def get_metrics(self, model, metrics_path = "metrics.csv"):
        AP             = 0.65
        size           = model.interpreter.get_input_details()[0]["shape"][1]
        max_detections = model.interpreter.get_output_details()[0]["shape"][1]
        df             = pd.read_csv("whole.csv")
        test_dataset   = df[df.iloc[:,0] == 'TEST']
        paths          = test_dataset.iloc[:,1].unique()

        mean_detection = []
        mean_iou       = []

        for path in tqdm(paths):        
            
            if "vertical_horizontal_fliped" in path: continue

            img   = cv2.imread(path)[:,:,::-1].astype(np.uint8)
            h,w,c = img.shape

            p0     = int(abs((h-w))/2)
            if h<w:
                pf     = int(w-p0)
                crop   = img[:,p0:pf,:].copy()
                labels = self.get_labels(test_dataset,path) 
                bb     = self.proportion_to_pixel(img.copy(),labels)
                bb[:,[1,3]] = bb[:,[1,3]]-p0

            elif h>w:
                pf     = int(h-p0)
                crop   = img[p0:pf,:,:].copy()
                labels = self.get_labels(test_dataset,path)
                bb     = self.proportion_to_pixel(img.copy(),labels)
                bb[:,[0,2]] = bb[:,[0,2]]-p0
            
            self.remove_bb_out_of_image(bb,h,w)

            if bb.shape[0] == 0:
                print("no labels", path)
                continue
            
            detection_frame = self.reshape(crop,size)
            detection_boxes, classes, scores = model.predict(detection_frame)
            bb_pred = self.proportion_to_pixel(crop,detection_boxes)

            metrics = []
            for t in np.linspace(0,1,51).round(3):
                metrics.append( 
                                self.get_IoUs(bb_pred,
                                                scores,
                                                bb,
                                                confidence_threshold = t,
                                                IoU_threshold        = AP,
                                                max_detections       = max_detections)
                               )

            mean_detection.append(np.array(metrics)[:,1])
            mean_iou.append(np.array(metrics)[:,0])

            print("detection:",np.array(metrics)[:,1].max(),"iou:",np.array(metrics)[:,0].max())

            thresholds = np.linspace(0,1,51).round(4)
            detection  = np.array(mean_detection).mean(axis =0).round(4)
            iou        = np.array(mean_iou).mean(axis =0).round(4)

            df = pd.DataFrame({
                                "thresholds": thresholds,
                                "detection":  detection,
                                "iou":        iou
                              })

            df.to_csv(metrics_path) 

        return iou, detection, thresholds  



   