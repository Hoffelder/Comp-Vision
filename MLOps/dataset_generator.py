import os 
from random import shuffle
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import random
import json

import os 
import json 
random.seed(42)
np.random.seed(42)

def draw_labels(labels,img,show = False):
    
    labels = np.array(labels).astype(int) 
    img = np.array(img).astype(np.uint8)  
    h,w,c  = img.shape
    for bb in labels:
        if max(bb)<=1:
            y = (bb[[1,3,5,7]]*h).astype(int)
            x = (bb[[0,2,4,6]]*w).astype(int)
            cv2.rectangle(img,(x[0],y[0]),(x[1],y[2]),(0,255,0),2)

        else:
            y = (bb[[1,3]]).astype(int)
            x = (bb[[0,2]]).astype(int)
            cv2.rectangle(img,(x[0],y[0]),(x[1],y[1]),(0,255,0),2)
        
    if show:
        plt.figure(figsize = (15,30))
        plt.imshow(img)
    
    return img

def read_json(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data


def parse_labelme(data,img):
    h,w,c = img.shape
    points = []
    
    def clip(value,top):
        return min(max(value,0),top)
    
    for shapes in data["shapes"]:
        x0,y0 = np.array(shapes["points"])[0]
        x1,y1 = np.array(shapes["points"])[1]

        if x0>x1:
            x1,x0 = x0,x1 

        if y0>y1:
            y1,y0 = y0,y1 
            
        if x1 == x0 or y1 == y0:
            continue

        x1,x0,y1,y0 = clip(x1,w), clip(x0,w), clip(y1,h), clip(y0,h)
        points.append([int(x0), int(y0), int(x1), int(y1)])
    
    return np.array(points)

def labelme_to_tf(data,img):
    h,w,c = img.shape
    points = []
    
    def clip(value):
        return min(max(value,0),1)
    
    for shapes in data["shapes"]:
        x0,y0 = np.array(shapes["points"])[0]
        x1,y1 = np.array(shapes["points"])[1]
        
        if x0>x1:
            x1,x0 = x0,x1 
        
        if y0>y1:
            y1,y0 = y0,y1 
        
        x1,x0,y1,y0 = clip(x1),clip(x0),clip(y1),clip(y0)
        
        if x1 == x0 or y1 == y0:
            continue

        points.append([x0/w, y0/h,  x1/w, y0/h, x1/w, y1/h, x0/w, y1/h])
       
    
    return np.array(points)
    
    
def proportion_to_pixel(img,labels):
    h,w,c  = img.shape
    bb = []
    for proportional_bb in labels:

        y = (proportional_bb[[1,3,5,7]]*h).astype(int)
        x = (proportional_bb[[0,2,4,6]]*w).astype(int)
        x_min, y_min, x_max, y_max = x[0],y[0],x[1],y[2]
        bb+=[[x_min, y_min, x_max, y_max]]

    return np.array(bb)
        
def pixel_to_proportion(img,labels):
    h,w,c  = img.shape
    bb = []
    for proportional_bb in labels:
            
        y = proportional_bb.astype(np.float32)[[0,2]]/h
        x = proportional_bb.astype(np.float32)[[0,2]]/w
        x_min, y_min, x_max, y_max = x[0],y[0],x[1],y[1]
        
        if x_min == x_max or y_max == y_min:
            continue
        
        bb+=[[x_min, y_min, x_max, y_max]]
        
    return np.array(bb)
    

class Change:
    def __init__(self):
        
        self.change_transform = A.Compose([

        A.OneOf([

            #A.HueSaturationValue (hue_shift_limit=130, sat_shift_limit=140, val_shift_limit=30, always_apply=False, p=0.99),

            #A.RandomBrightnessContrast (brightness_limit=0.3, contrast_limit=0.05, brightness_by_max=True, always_apply=True, p=0.99)

            A.Blur(blur_limit=10, p=0.99)

            ], p=1)

        ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))


    def apply(self,img,bboxes):
        return self.change_transform(image=img, bboxes=bboxes)


class Noise:
    def __init__(self):

        self.noise_transform = A.Compose([

        A.OneOf([

            #A.ImageCompression (quality_lower=65, quality_upper=75, always_apply=False, p=0.99), 

            #A.ISONoise(color_shift=(0.2, 0.3), intensity=(0.1, 0.2), always_apply=False, p=0.99), 

            #A.MultiplicativeNoise (multiplier=(0.7, 1.3), per_channel=True, elementwise=False, always_apply=True, p=0.99),

            A.Spatter(mean=0.65, 
                          std=0.5, 
                          gauss_sigma=2, 
                          cutout_threshold=0.68, 
                          intensity=0.6, 
                          mode='rain', 
                          always_apply=True, 
                          p=0.99),

            A.Sharpen(alpha=(0.2, 0.25), lightness=(2, 4), always_apply=True, p=0.99),
            
            A.RGBShift(r_shift_limit=130, g_shift_limit=130, b_shift_limit=130, always_apply=True, p=0.99)

            ], p=1) ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
        
    def apply(self,img,bboxes):
        
        return self.noise_transform(image=img, bboxes=bboxes) 


class Color:
    def __init__(self):
        
        self.color_transform = A.Compose([

        A.OneOf([

            #A.HueSaturationValue (hue_shift_limit=130, sat_shift_limit=140, val_shift_limit=30, always_apply=False, p=0.99),

            #A.RandomBrightnessContrast (brightness_limit=0.3, contrast_limit=0.05, brightness_by_max=True, always_apply=True, p=0.99)

            A.RandomGamma (gamma_limit=(170, 280), eps=None, always_apply=True, p=0.99)

            ], p=1)

        ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))


    def apply(self,img,bboxes):
        
        return self.color_transform(image=img, bboxes=bboxes)
    

class Crop:
    def __init__(self,crop_proportions = None):
        self.iters = 0
        
        if crop_proportions is None:
            #self.crop_proportions     = [0.15, 0.3, 0.5, 0.6]
            #self.crop_proportions_4k  = [0.15, 0.3, 0.5, 0.6]
            #[i for i in np.linspace( 0.1,0.8,6)]

            alpha = 0.2
            self.crop_proportions_4k = (np.random.rand(2)*(1-2*alpha)+alpha).tolist()
            self.crop_proportions    = (np.random.rand(2)*(1-2*alpha)+alpha).tolist()
            
    def apply(self,img,bboxes,previous_transforms):
        h,w,c  = img.shape
        self.crops = {}
        
        if h > 2000 and w > 2000:
            
            for proportion in self.crop_proportions_4k :
                
                visibility = 0.95 if proportion < 0.31 else 0.05 
                
                self.crops[str(round(proportion,3))] = A.Compose([
                    A.RandomCrop(width=int(w*proportion), height=int(h*proportion))
                ],bbox_params = A.BboxParams(format='pascal_voc', min_visibility = visibility, label_fields=[]))
        else:  
            for proportion in self.crop_proportions:
                
                visibility = 0.95 if proportion < 0.31 else 0.05 
                
                self.crops[str(round(proportion,3))] = A.Compose([
                    A.RandomCrop(width=int(w*proportion), height=int(h*proportion))
                ],bbox_params = A.BboxParams(format='pascal_voc', min_visibility = visibility, label_fields=[]))
        
        
        transformations = {}
        
        for proportion,crop in self.crops.items():

            croped = crop(image=img, bboxes=bboxes)
            
            tries = 0
            while len(croped["bboxes"])<3 and tries <= 200:
                croped = crop(image=img, bboxes=bboxes)
                tries+=1
            
            if tries==201:
                print("crop tries exceed 200")
                continue
                
            transformations[previous_transforms+"+crop:v"+str(self.iters)+"-p"+str(proportion)] = {"image":croped["image"],"bboxes":croped["bboxes"]}
                
            self.iters+=1
            
        return transformations
            

class Rotations:
    def __init__(self):
        
        self.vertical = A.Compose([

            A.VerticalFlip(p=1)

        ],bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.05, label_fields=[]))

        self.horizontal = A.Compose([

            A.HorizontalFlip(p=1)

        ],bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.05, label_fields=[]))


        self.vertical_horizontal = A.Compose([

            A.VerticalFlip(p=1),
            A.HorizontalFlip(p=1)

        ],bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.05, label_fields=[]))

    def apply(self,img,bboxes,root):
        
        transformations = {}
        
        print("TOTAL LABELS:",np.array(bboxes).shape)
        
        transformations[root+"_RAW"] = {"image":img.copy(),"bboxes":np.array(bboxes).astype(int).tolist()}
        
        #vertical_fliped    = self.vertical(image=img, bboxes=bboxes)
        #transformations[root+"_vertical_fliped"] = {"image":vertical_fliped["image"],"bboxes":np.array(vertical_fliped["bboxes"]).astype(int).tolist()}
        
        #vertical_fliped    = self.horizontal(image=img, bboxes=bboxes)
        #transformations[root+"_horizontal_fliped"] = {"image":vertical_fliped["image"], "bboxes":np.array(vertical_fliped["bboxes"]).astype(int).tolist()}
        
        vertical_fliped    = self.vertical_horizontal(image=img, bboxes=bboxes)
        transformations[root+"_vertical_horizontal_fliped"] = {"image":vertical_fliped["image"],"bboxes":np.array(vertical_fliped["bboxes"]).astype(int).tolist()}
        
        return transformations


def convert_to_tf_lite(image,bboxes,image_path):
    '''
    (x1,y1) ---------------  (x2,y2)
      |                         |
      |                         |
      |                         |
      |                         |
    (x3,y3) ---------------- (x4,y4)

    '''

    bboxes  = bboxes
    h, w, c = image.shape
    
    tf_bounding_boxes = []
    for bb in bboxes:
        x_min, y_min, x_max, y_max = bb
        
        #            "x1",        "x2",          "x3",        "x4",
        #bb = [float(x_min), float(x_max), float(x_min), float(x_max), 
        #       "y1",              "y2",        "y3",          "y4"
        #      float(y_min), float(y_min), float(y_max), float(y_max)]
        
        
         #            "x1",        "y1",                "x2",        "y1",
        bb = [float(x_min), float(y_min),   float(x_max), float(y_min), 
        #           "x2",       "y2",          "y3",          "y4"
              float(x_max), float(y_max),   float(x_min), float(y_max)]

        bb = np.array(bb).astype(np.float16)
        
        bb[[1,3,5,7]] = (bb[[1,3,5,7]]/h).astype(np.float16)
        bb[[0,2,4,6]] = (bb[[0,2,4,6]]/w).astype(np.float16)
        
        
        tf_bounding_boxes.append(bb)
        #["split", "path", "log","x1","x2","x3","x4","y1","y2","y3","y4"]

    return np.array(tf_bounding_boxes)


def AugumentationPipeline(label_paths,data_integrity,show = False):
    
    image_path = label_paths.replace(".json",".jpg")
    root       = label_paths.replace(".json","").replace("/","_")

    if not os.path.exists(image_path):
        image_path = image_path.replace(".jpg",".png")
    
        if not os.path.exists(image_path):
            image_path = image_path.replace(".png",".JPG")

    img    = cv2.imread(image_path)
    
    data_integrity[image_path] = {}
    data_integrity[image_path]['img_shape'] = img.shape

    data   = read_json(label_paths)
    labels = parse_labelme(data,img)
    bboxes = pixel_to_proportion(img,labels)

    data_integrity[image_path]['labels'] = len(bboxes)

    augmentations = {}
    rotations     = Rotations()
    crop          = Crop()
    color         = Color()
    change        = Change()
    noise         = Noise()
    
    rotated       = rotations.apply(img,labels,root)
    
    print("applying rotations")
    augmentations.update(rotated)
    
    print("applying crop")
    
    for key in rotated: 
        crop_transforms = crop.apply(rotated[key]["image"],rotated[key]["bboxes"],key)
        augmentations.update(crop_transforms)
    
    keys = list(augmentations.keys())
    
    print("applying change color and noise augumentations")
    
    for key in keys:
        
        transformed_color  = color.apply(augmentations[key]["image"] ,augmentations[key]["bboxes"])
        transformed_change = change.apply(augmentations[key]["image"],augmentations[key]["bboxes"])
        transformed_noise  = noise.apply(augmentations[key]["image"] ,augmentations[key]["bboxes"])

        augmentations[key+"+color"]  = transformed_color
        augmentations[key+"+change"] = transformed_change
        augmentations[key+"+noise"]  = transformed_noise
    
        if show:
            show_detections(augmentations,transformed_color,key)
            
    save_path = "augumentations/"+label_paths.replace(".json","").split("/")[1]
    
    if not os.path.exists("augumentations"): 
        os.mkdir("augumentations")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    df_dict = {
        "split": [],
        "path":  [],
        "log":   [],
        "x1":    [],
        "x2":    [],
        "x3":    [],
        "x4":    [],
        "y1":    [],
        "y2":    [],
        "y3":    [],
        "y4":    []
    } 
    
    print("saving images and labels")
    
    prob = np.random.rand()
    
    if prob < 0.7:
        split = "TRAIN"   
    elif prob < 0.9:
        split = "TEST" 
    else: 
        split = "VALIDATION"
    
    for key in list(augmentations.keys()):
        
        image_path = save_path + "/" + key+ ".jpg"
        h,w,c      = augmentations[key]["image"].shape
        tf_lite_bb = convert_to_tf_lite(augmentations[key]["image"],augmentations[key]["bboxes"],image_path)
        
        if len(tf_lite_bb)>300:
            print("too many labels in image",len(tf_lite_bb))
            continue

        for bb in tf_lite_bb:
            df_dict["split"].append(split)
            df_dict["path"].append(image_path)
            df_dict["log"].append("log") 
            df_dict["x1"].append(bb[0])  
            df_dict["x2"].append(bb[1])   
            df_dict["x3"].append(bb[2])   
            df_dict["x4"].append(bb[3])   
            df_dict["y1"].append(bb[4])   
            df_dict["y2"].append(bb[5])   
            df_dict["y3"].append(bb[6])   
            df_dict["y4"].append(bb[7]) 
        
        cv2.imwrite(image_path, augmentations[key]["image"])
        
    return pd.DataFrame(df_dict)   
        

def generate_all_label_paths():
    
    folders =   ['raw/Eucalipto',
    		     'raw/Arcelor',
                 'raw/Carrinhos',
                 'raw/CENIBRA_3',
                 'raw/Eucalipto',
                 'raw/Oldpinus',
                 'raw/remasa_ibema26.11'] #'raw/Pinus'

    label_paths = []

    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            if ".json" in file:
                label_paths.append(folder+"/"+file)

        
    return label_paths


if __name__ == "__main__":
    df     = None
    labels = generate_all_label_paths()
    shuffle(labels)

    data_integrity = {}
    for i,label in enumerate(labels[:3]):
        try:
            print("*************************************************************************************")
            print(round(i/len(labels)*100,2),"%"," Augumenting image:",i,"out of", len(labels),"completed", "path",label)
            print("*************************************************************************************")

            if df is None:
                df = AugumentationPipeline(label,data_integrity,show = False)
            else:
                df2 = AugumentationPipeline(label,data_integrity,show = False)
                df = pd.concat([df,df2])
            df.to_csv("dataset_versioning/1.0.0.csv",header=False, index=False)
        except Exception as e:
            print(e)

    print(data_integrity)
    with open('dataset_versioning/1.0.0.json', 'w') as fp:
        json.dump(data_integrity, fp,  indent=4)


