import json 
import cv2
from tqdm import tqdm


def data_integrity_check(experiment_name):
    f = open("dataset_versioning/"+ experiment_name +".json")
    integrity = json.load(f)
    
    for key in tqdm(integrity):
        img = cv2.imread(key)
        f = open(key.replace(".png",".json").replace(".jpg",".json").replace(".JPG",".json"))
        labels = json.load(f)

        assert tuple(integrity[key]["img_shape"]) == img.shape  , "different images shapes"
        assert integrity[key]["labels"] == len(labels["shapes"]), "different number of labels"

    return True

data_integrity_check("1.0.0")