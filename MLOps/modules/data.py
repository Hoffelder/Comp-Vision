from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

import itertools
from os.path import exists
import json


class Transform:
    def __init__(self) -> None:
        pass
    def fit(self,X_train):
        self.scale.fit(X_train)

    def normalize(self,X_train,X_test):
        self.scale = StandardScaler()
        self.fit(X_train)

        return self.scale.transform(X_train),self.scale.transform(X_test)
    
    def split(self,X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        return X_train, X_test, y_train, y_test

class Tflite:

    def __init__(self,experiment_name) -> None:
        self.training_status = None
        self.experiment_name = experiment_name

    def generate_train_parameters_file(self):
        
        params = {   
            "epochs": [50],
            "batch_size": [8,16,32],
            "steps_per_execution": [1],
            "tflite_max_detections": [100,500],
            "var_freeze_expr": ['(efficientnet|fpn_cells|resample_p6)']
        }
        
        hiper_parameters = []
        parameters_names = []
        for parameter_name in params:
            parameters_names.append( parameter_name )
            hiper_parameters.append( tuple(params[parameter_name]))

        parameter_values = (list(itertools.product(*hiper_parameters)))

        training = {}

        for i,p in enumerate(parameter_values):
            training[str(i)] = {name:param for name,param in zip(parameters_names,p)}
            training[str(i)]["Done"] = "false"

        file_exists = exists("hiper_parameters/" + self.experiment_name + ".json")
        if not file_exists:
            with open("hiper_parameters/" + self.experiment_name + ".json", "w") as outfile:
                json_object = json.dumps(training,indent=4)
                outfile.write(json_object)

    def read_train_parameters_file(self):
        with open("hiper_parameters/" + self.experiment_name + ".json", "r") as openfile:
            self.training_status = json.load(openfile)

        return self.training_status


class Data:
    def __init__(self,experiment_name) -> None:
        self.transform = Transform()
        self.tflite    = Tflite(experiment_name)
    
    def generate(self):
        self.tflite.generate_train_parameters_file()
        return self.tflite.read_train_parameters_file()
        

