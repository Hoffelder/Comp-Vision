from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import tensorflow as tf
assert tf.__version__.startswith('2')
import sys
from tensorflow.python.client import device_lib
import numpy as np
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tensorflow.python.client import device_lib
from absl import logging

from os.path import exists
import os

device_lib.list_local_devices()
tf.test.gpu_device_name()
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)


import numpy as np
import tensorflow as tf

class EfficientDet:
    def __init__(self, path = "models/0.0.1.tflite") -> None:
        # Location of tflite model file (int8 quantized)
        self.model_path = path
        # Load TFLite model and allocate tensors.
        self.interpreter     = tf.lite.Interpreter(model_path=self.model_path)
        self.input_details   = self.interpreter.get_input_details()
        self.output_details  = self.interpreter.get_output_details()
        # Allocate tensors
        self.interpreter.allocate_tensors()
        outname              = self.output_details[0]['name']
        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            self.boxes_idx   = 1
            self.classes_idx = 3
            self.scores_idx  = 0
        else: # This is a TF1 model
            self.boxes_idx   = 0
            self.classes_idx = 1
            self.scores_idx  = 2

    def predict(self,frame):
        # Create input tensor out of raw features
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        # Run inference
        self.interpreter.invoke()
        boxes   = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0]   # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0] # Class index of detected objects
        scores  = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0]  # Confidence of detected objects    
        return boxes, classes, scores


class Models:
    def __init__(self) -> None:
    
        self.models = {
            "efficientdet-lite0":     object_detector.EfficientDetSpec
        }
    
    def instantiate(self,model_name,parameter): 

        model = self.models[model_name](**parameter)

        return model

    def instantiate(self,**params):
       
        #EFFICIENTDET LITE 0
        #, 'autoaugment_policy' : policy_v0()
        #spec = object_detector.EfficientDetSpec(
        #    model_name            = 'efficientdet-lite0',
        #    uri                   = 'https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
        #    hparams               = {'max_instances_per_image' : 300,
        #                                     'optimizer' : 'adam', 
        #                                     'learning_rate' : 0.001, 
        #                                     'lr_warmup_init' : 0.0001},
        #    model_dir             = './models',
        #    epochs                = params["epochs"],
        #    batch_size            = params["batch_size"],
        #    steps_per_execution   = 1,
        #    moving_average_decay  = 0,
        #    #var_freeze_expr = '(efficientnet|fpn_cells|resample_p6)',
        #    var_freeze_expr       = params["var_freeze_expr"],
        #    tflite_max_detections = params["tflite_max_detections"],
        #    strategy              = None,
        #    tpu                   = None,
        #    gcp_project           = None,
        #    tpu_zone              = None,
        #    use_xla               = False,
        #    profile               = False,
        #    debug                 = False,
        #    tf_random_seed        = 111111,
        #    verbose               = 1
        #)

        spec = object_detector.EfficientDetSpec(
            model_name            = 'efficientdet-lite1',
            uri                   = 'https://tfhub.dev/tensorflow/efficientdet/lite1/feature-vector/1',
            hparams               = {'max_instances_per_image' : 500,
                                             'optimizer' : 'adam', 
                                             'learning_rate' : 0.001, 
                                             'lr_warmup_init' : 0.0001},
            model_dir             = './models',
            epochs                = params["epochs"],
            batch_size            = params["batch_size"],
            steps_per_execution   = 1,
            moving_average_decay  = 0,
            #var_freeze_expr = '(efficientnet|fpn_cells|resample_p6)',
            var_freeze_expr       = params["var_freeze_expr"],
            tflite_max_detections = params["tflite_max_detections"],
            strategy              = None,
            tpu                   = None,
            gcp_project           = None,
            tpu_zone              = None,
            use_xla               = False,
            profile               = False,
            debug                 = False,
            tf_random_seed        = 111111,
            verbose               = 1
        )



        train_data, validation_data, test_data = object_detector.DataLoader.from_csv('whole.csv')

        model = object_detector.create(train_data, model_spec = spec,
                                    train_whole_model      = True, 
                                    validation_data        = validation_data)

        return model,test_data

    def save(self,model,name,path):
        
        if not exists(path):
            os.mkdir(path)

        model.export(export_dir= path +'/' + name)



