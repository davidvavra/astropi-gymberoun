import os
import warnings
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from fileOne import data_generator
#from tensorflow.keras.models import load_model
from keras_applications import imagenet_utils
from PIL import Image
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from deeplab_v3_plus import DeepLabV3Plus
from test_model import read_image

MODEL_NAME = "models/model-01.hdf5"
DATA_DIR = "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset"      # Where are my data ? XD

backend = tf.keras.backend
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
# XLA Flags 
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/cuda'
gpus = tf.config.experimental.list_physical_devices('GPU')
    # Set memory growth to True 
    # (will prevent some fails if low memory and improve memory management while slightly slowing progress) 
    # - stability imporvement
tf.config.experimental.set_memory_growth(gpus[0], True)
"""
tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
logical_gpus = tf.config.list_logical_devices('GPU')
print(logical_gpus)
"""

def categorical_crossentropy_with_logits(out_original, out_prediction):
    # compute cross entropy
    cross_entropy = backend.categorical_crossentropy(out_original, out_prediction, from_logits=True)

    # compute loss
    loss = backend.mean(cross_entropy, axis=-1)

    return loss

def load_image(name):
    image= Image.open(name)
    #print(img)
    #print('Load Image', name, np.array(img).shape)
    return np.array(image)

# TF_GPU_ALLOCATOR=cuda_malloc_async

if __name__ == "__main__":
    print(MODEL_NAME)
    # List all validation data (images & masks)
    images = sorted(glob(os.path.join(DATA_DIR, "validation/images/*")))
    masks = sorted(glob(os.path.join(DATA_DIR, "validation/masks/*")))

    # Creating training and validation datasets
    #dataset = data_generator(images, masks, idk=True)

    saved_tflite_model_path = "Saved_TFLite_Model"

    if not os.path.isdir(saved_tflite_model_path):
        os.makedirs(saved_tflite_model_path)

    filename=MODEL_NAME

    if os.path.isfile(filename):
        print('Model ', filename, ' exists')

        """model = tf.keras.models.load_model(filename, , custom_objects={
            'categorical_crossentropy_with_logits': categorical_crossentropy_with_logits})"""
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model = DeepLabV3Plus(num_classes=10, base_model="MobileNetV2")(input_size=(1024,1024))
        model.compile(
            optimizer=RMSprop(learning_rate=1e-3),
            loss=loss,
            metrics=["accuracy"],
        )
        model.load_weights(MODEL_NAME)

        model.summary()
        print(model.get_layer("input_1").input_shape)

        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
        converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
        converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
        converter_float32_ = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name



        height = 1024
        width = 1024

        for i, name in enumerate(images):

            image=load_image(name)
            image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last',
                                                    mode='torch')
            image=cv2.resize(image,(width,height),interpolation=cv2.INTERSECT_NONE)
            image = np.expand_dims(image, axis=0)
            dataset_ = tf.data.Dataset.from_tensor_slices((image)).batch(1)

        
        def representative_data_gen():
            print("REP DATASET CALLED")
            for input_value in dataset_.take(10):
                print([input_value])
                yield [input_value]

    model_quant_file="q_" + MODEL_NAME.split("/")[-1].split(".")[0]
    
    #### INT8 QUANTIZATION #######

    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.target_spec.supported_types=[tf.compat.v1.lite.constants.FLOAT16]
    converter_int8.inference_input_type = tf.uint8
    converter_int8.inference_output_type = tf.uint8
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_data_gen
    tflite_model_quant_INT8 = converter_int8.convert()

    tflite_model_quant_file_INT8 = model_quant_file+'_INT8' +'.tflite'
    print(tflite_model_quant_file_INT8)
    tflite_model_quant_path_INT8 = os.path.join(saved_tflite_model_path,tflite_model_quant_file_INT8)
    print(tflite_model_quant_path_INT8)
    open(tflite_model_quant_path_INT8, "wb").write(tflite_model_quant_INT8)
    print('Conversion Successful. File written to ', tflite_model_quant_path_INT8)