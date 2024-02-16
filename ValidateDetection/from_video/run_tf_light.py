import numpy as np
import tensorflow as tf

class Model:
    def __init__(self,model_path) -> None:
        # Location of tflite model file (int8 quantized)
        self.model_path = model_path
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


if __name__ == "__main__":

    # Add dimension to input sample (TFLite model expects (# samples, data))
    np_features = np.expand_dims(np.zeros((320,320,3))+200, axis=0).astype(np.uint8)

    print(np_features.shape)
    model = Model()

    boxes, classes, scores = model.predict(np_features)

    print("boxes",  boxes)
    print("classes",classes)
    print("scores", scores)
