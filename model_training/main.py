# Import libraries
import datetime
import os
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import tensorboard
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from deeplab_v3_plus import DeepLabV3Plus
from fileOne import data_generator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow_examples.models.pix2pix import pix2pix

# Tensorflow log level 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
# XLA Flags 
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/cuda'
# Matplotlib settings
matplotlib.use('TkAgg')

# CONFIG DATA
IMAGE_SIZE = 1024                                                        # Image resolution - 800 - 1200
NUM_CLASSES = 10                                                        # How many classes ?
BATCH_SIZE = 1                                                          # How many image to process at one time
EPOCHS = 100                                                             # How many training epochs ?
DATA_DIR = "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset"      # Where are my data ? XD

if __name__ == "__main__":
    """
        Preparation
    """
    # Find out all GPUs 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Set memory growth to True 
    # (will prevent some fails if low memory and improve memory management while slightly slowing progress) 
    # - stability imporvement
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # Create model
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    #model = DeeplabV3Plus_m1(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    #model = unet(img_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    #model = g_unet(img_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    #model = models.DeepLabV3Plus(num_classes = NUM_CLASSES)(input_size=IMAGE_SIZE)

    model = DeepLabV3Plus(num_classes=NUM_CLASSES, base_model="MobileNetV2")(input_size=(IMAGE_SIZE, IMAGE_SIZE))

    #model.summary() 
    #model = models.unet_3plus_2d((IMAGE_SIZE, IMAGE_SIZE,3), NUM_CLASSES, filter_num_down=[64, 128, 256, 512, 1024], filter_num_skip=[64, 64, 64, 64], stack_num_down=10, stack_num_up=10, backbone='ResNet50' )
    # Print model summary
    model.summary()
    x = input("Do you want to continue ? [y/n]")
    if(x != "y"):
        exit()

    """
        Dataset creation
    """

    # List all training data (images & masks)
    train_images = sorted(glob(os.path.join(DATA_DIR, "train/images/*")))
    train_masks = sorted(glob(os.path.join(DATA_DIR, "train/masks/*")))

    # List all validation data (images & masks)
    val_images = sorted(glob(os.path.join(DATA_DIR, "validation/images/*")))
    val_masks = sorted(glob(os.path.join(DATA_DIR, "validation/masks/*")))

    # Creating training and validation datasets
    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    """
        ML training
    """

    # Setup Sparce Categorical Crossentrropy loss function
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Compile model
    model.compile(
        # using Adam (Adaptive Moment Estimation) optimizer with learning rate of 0.1%
        #optimizer=Adam(learning_rate=0.00001),
        optimizer=RMSprop(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )

    # Setup for tensorboard for easier data reviewing
    file_name = "data_log-" + str(int(round(datetime.datetime.now().timestamp())))
    tensorboard = TensorBoard(log_dir="logs//{}".format(file_name))
    
    outputFolder = './models'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    filepath=outputFolder+"/model-{epoch:02d}-{val_accuracy:.2f}.hdf5"


    checkpoint_callback = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1,
    save_best_only=True, save_weights_only=False,
    save_frequency=5)


    # Train model of train dataset and validate on validation dataset
    # epochs count from variable
    # callback for tensorboard to save data
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    model.save(f'saved_model/my_model-{str(int(round(datetime.datetime.now().timestamp())))}')
    # Optional (if 1)
    # Will show graphs of training statistic, also visible from tensorboard
    if True:
        figure, axis = plt.subplots(2, 2, figsize=(16,12), dpi=80)

        figure.tight_layout(h_pad=3)
        figure.suptitle('Barrande AI training')
        plt.subplots_adjust(top=0.95)
    
        axis[0,0].plot(history.history["loss"])
        axis[0,0].set_title("Training Loss")
        axis[0,0].set_ylabel("loss")
        axis[0,0].set_xlabel("epoch")

        axis[0,1].plot(history.history["accuracy"])
        axis[0,1].set_title("Training Accuracy")
        axis[0,1].set_ylabel("accuracy")
        axis[0,1].set_xlabel("epoch")

        axis[1,0].plot(history.history["val_loss"])
        axis[1,0].set_title("Validation Loss")
        axis[1,0].set_ylabel("val_loss")
        axis[1,0].set_xlabel("epoch")

        axis[1,1].plot(history.history["val_accuracy"])
        axis[1,1].set_title("Validation Accuracy")
        axis[1,1].set_ylabel("val_accuracy")
        axis[1,1].set_xlabel("epoch")

        plt.show()