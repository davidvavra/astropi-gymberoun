"""
Written for ESA's astropi challenge 2022/2023
Team: Barrande

Code for creating dataset and training, testing, converting and evaluating models for semantic images segmentation.
Models: 
- Deeplab V3 plus (two implementations)
- UNet (two implementations)
- FCN
- PAN

@Author: Cyril Šebek
@Github: https://github.com/Blboun3
@Project: https://github.com/davidvavra/astropi-gymberoun
"""
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from glob import glob

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import imagenet_utils

# Modules written by me
from modules.myDLABV3_plus import DeeplabV3Plus as mDLAB
from modules.myUNET import UNET
from modules.prep_data import main as prep_data
from modules.utils import *

"""
Modules by:
@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation
"""
from modules.deeplab_v3_plus import DeepLabV3Plus
from modules.fcn import FCN
from modules.pan import PAN
from modules.unet import UNet

parser = argparse.ArgumentParser(
    description = 'Astropi AI creation tool for semantic segmentation models\nTool for creating dataset, creating, evaluating and testing semantic segmentation models and model conversion to tflite INT8 fully quantized model almost ready to be run on EdgeTPU',
    epilog = "Created for ESA's astropi challenge 2022/2023 for team Barrande by Cyril Šebek ; github: https://github.com/Blboun3 ; astropi project: https://github.com/davidvavra/astropi-gymberoun"
)

parser.add_argument('--mode', help="Choose which mode to use.", type=str, required=True, choices=["dataset", "train", "evaluate", "convert", "test"])
parser.add_argument('--color_conversion_map', help="Only for 'dataset' mode. Path to file with color conversion map.", type=str, default="color_conversion_map.txt", required=False)
parser.add_argument('--dataset_max_thread', help="Maximum thread to run when creating dataset. Only for mode 'dataset'", type=int, default=15, required=False)
parser.add_argument('--train_epochs', help="How many training epochs to run. Only for 'train'.", type=int, default=20, required=False)
parser.add_argument('--image_size', help="Image must be square, size of side in pixels. For all modes except 'dataset' and 'convert'.", default=1024, type=int, required=False)
parser.add_argument('--batch_size', help="How big should batch_size be. For modes: 'evaluate', 'train', 'convert'", default=1, type=int, required=False)
parser.add_argument('--test_image', help='If test mode is selected show inference with this image (image will be shown)', type=str, required=False)
parser.add_argument('--model_path', help='Path to saved model. Only for modes [\'test\',\'evaluate\',\'convert\']', required=False, type=str)
parser.add_argument('--save_path', help="Path to folder where models should be saved. Only for 'training' or 'convert'.", required=False, type=str, default='models')
parser.add_argument('--conversion_size', help='Size of representative dataset for conversion, remember, conversion to tflite with full INT8 quantization is pretty lenghty process, recommended value is ~100 images. Will be selected randomly from validation set, -1 will use all images from validation, -2 will use all images. Only if mode \'convert\'', default=100, type=int, required=False)
parser.add_argument('--expand_dataset', help="Automatically expand dataset. Good if dataset is too small. Only if mode is 'dataset'", type=str2bool, default=True, required=False)
parser.add_argument('--model', help="Choose semantic segmentation model to use.", type=str, required=False, default='UNET', choices=["UNET", "UNet", "DeepLabV3_plus", "DLABV3_plus", "FCN", "PAN"])
parser.add_argument('--backbone', help="Choose backone model.", type=str, default=None, required=False, choices=["VGG16", "VGG19", "MobileNetV1", "MobileNetV2"])
parser.add_argument('--num_classes', help="The number of classes to be segmented.", default=10, type=int, required=False)
parser.add_argument('--optimizer', help="Optimizer to use for the model.", type=str, choices=['Adam', 'Adagrad', 'SGD'], required=False)
parser.add_argument('--base_dataset_dir', help="Base folder where the images (dataset) are located.", type=str, required=True)
parser.add_argument('--colormap', help="Path to file containing colormap to recolor images after inferencing. Only if mode is 'test'", required=False, type=str, default="colormap.txt")

args = parser.parse_args()

# Matplotlib won't work on my computer otherwise, feel free to change it if other setting works for you
mpl.use('TkAgg')

# Load parameters from user
IMAGE_SIZE = args.image_size
NUM_CLASSES = args.num_classes
BATCH_SIZE = args.batch_size
DATA_DIR = args.base_dataset_dir
if(args.mode == 'train'):
    MODE = 1
elif(args.mode == 'test'):
    MODE = 2
elif(args.mode == 'convert'):
    MODE = 3
elif(args.mode == 'evaluate'):
    MODE=4
elif(args.mode == 'dataset'):
    MODE = 5
MODEL = args.model_path

# Create datasets
# List all training data (images & masks)
train_images = sorted(glob(os.path.join(DATA_DIR, "train/images/*")))
train_masks = sorted(glob(os.path.join(DATA_DIR, "train/masks/*")))

# List all validation data (images & masks)
val_images = sorted(glob(os.path.join(DATA_DIR, "validation/images/*")))
val_masks = sorted(glob(os.path.join(DATA_DIR, "validation/masks/*")))

# Creating training and validation datasets
train_dataset = data_generator(train_images, train_masks, image_size=args.image_size)
val_dataset = data_generator(val_images, val_masks, image_size=args.image_size)
combined_dataset = data_generator(train_images+val_images, train_masks+val_masks, image_size=args.image_size)

# List of availible GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
# Set memory growth to True 
# (will prevent some fails if low memory and improve memory management while slightly slowing progress) 
# - stability imporvement
tf.config.experimental.set_memory_growth(gpus[0], True)
# Create model
# Free up RAM in case the model definition cells were run multiple times or there are any leftover trash in GPU RAM
tf.keras.backend.clear_session()

def create_model():
    """Function to create, build and compile model

    Returns:
        tf.keras.Model: Build and compiled model
    """
    model = args.model
    backbone = args.backbone
    optimizer = args.optimizer

    if(model == 'UNET'):
        model = UNET(img_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    elif(model == 'UNet'):
        model = UNet(NUM_CLASSES, base_model=backbone)(input_size=(IMAGE_SIZE, IMAGE_SIZE))
    elif(model == 'DeepLabV3_plus'):
        model = DeepLabV3Plus(NUM_CLASSES, version='DeepLabV3Plus', base_model=backbone)(input_size=(IMAGE_SIZE, IMAGE_SIZE))
    elif(model == "DLABV3_plus"):
        model = mDLAB(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    elif(model == "FCN"):
        model= FCN(NUM_CLASSES, base_model='MobileNetV2')(input_size=(IMAGE_SIZE, IMAGE_SIZE))
    elif(model == "PAN"):
        model = PAN(NUM_CLASSES, base_model="MobileNetV2")(input_size=(IMAGE_SIZE, IMAGE_SIZE))
    
    if(optimizer == 'Adam'):
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
    elif(optimizer == 'Adagrad'):
        tf.keras.optimizers.Adagrad(lr=0.001)
    elif(optimizer == 'SGD'):
        tf.keras.optimizers.SGD(lr=0.001)

    model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])

    return model


if __name__ == "__main__" and MODE == 1:
    """Training mode
    """
    # Create model
    model = create_model()
    # Create callback for saving checkpoints
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        f'{args.save_path}/checkpoint.hdf5', monitor='val_accuracy', verbose=1,
        save_best_only=True, save_weights_only=False,
        save_frequency=5)
    # Print model summary
    model.summary()
    # Ask if everything is OK
    x = input("Continue ? [y/n]")
    if(x != "y"):
        exit()

    # If so continue
    # Train model
    model.fit(train_dataset, epochs=20, callbacks=[checkpoint_callback], validation_data=val_dataset, shuffle=True)

elif __name__ == "__main__" and MODE == 2:
    """Testing mode
    """
    # Create model
    model = create_model()
    # Load model weights from file
    model.load_weights(MODEL)
    # Print model summary
    model.summary()
    # Run inference and plot predictions
    plot_predictions([args.test_image], load_colormap(args.colormap), model=model)

elif __name__ == "__main__" and MODE == 3:
    # Conversion mode

    saved_tflite_model_path = args.save_path

    if not os.path.isdir(saved_tflite_model_path):
        os.makedirs(saved_tflite_model_path)

    filename=MODEL

    if os.path.isfile(filename):
        # Create and load model
        model = create_model()
        model.load_weights(MODEL)
        model.summary()
        # Create converter
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name

    # Representative dataset generator
    def representative_data_gen():
        if(args.conversion_size == -1):
            for image,mask in val_dataset:
                yield [image]
        elif(args.conversion_size == -2):
            for image,mask in combined_dataset:
                yield [image]
        else:
            for image,mask in val_dataset.shuffle(buffer_size=128).take(args.conversion_size):
                yield [image]

    model_quant_file="q_" + MODEL.split("/")[-1].split(".")[0]

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

elif __name__ == '__main__' and MODE == 4:
    """Evaluation mode
    """
    # Create and load model 
    model = create_model()
    model.load_weights(MODEL)
    model.summary()
    print("Evaluating on all data")
    result = model.evaluate(combined_dataset, batch_size=BATCH_SIZE)
    print("Evaluation loss, acc:", result)

elif __name__ == '__main__' and MODE == 5:
    cc = args.color_conversion
    prep_data([os.path.join(DATA_DIR, "train/images"), os.path.join(DATA_DIR, "validation/images")], [[os.path.join(DATA_DIR, "train/masks"), os.path.join(DATA_DIR, "validation/masks")],val_dataset], max_threads=args.dataset_max_thread, rotate=args.expand_dataset, mask_recolor_map=get_recolor_map(args.color_conversion_map))