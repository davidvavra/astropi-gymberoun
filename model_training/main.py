# Import libraries
import datetime
import os
from glob import glob
from turtle import down
from unittest import skip

import matplotlib
import matplotlib.pyplot as plt
from pandas import concat
import tensorboard
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from tensorflow_examples.models.pix2pix import pix2pix
#from keras_unet_collection import models

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

def g_unet(img_size, num_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=[img_size,img_size,3], include_top=False)

    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project'
    ]
    
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512,3),
        pix2pix.upsample(256,3),
        pix2pix.upsample(128,3),
        pix2pix.upsample(64,3)
    ]

    inputs = tf.keras.layers.Input(shape=[img_size, img_size, 3])

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    
    last = tf.keras.layers.Conv2DTranspose(
        filters=num_classes, kernel_size=3, strides=2, padding='same'
    )

    #x = tf.keras.layers.Dropout(0.2)(x)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def unet(img_size, num_classes):
    img_size = (img_size, img_size)
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        #x = layers.Dropout(0.2)(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.Dropout(0.2)(x)
    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        #x = layers.Dropout(0.)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.Dropout(0.2)(x)
    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    """Block with one convolutional layer of type Conv2D with all necesities around it (activation, batchNormalization, etc.)

    Args:
        block_input (Tensor): Input
        num_filters (int, optional): Pass to layers.Conv2D. Defaults to 256.
        kernel_size (int, optional): Pass to layers.Conv2D. Defaults to 3.
        dilation_rate (int, optional): Pass to layers.Conv2D. Defaults to 1.
        padding (str, optional): Pass to layers.Conv2D. Defaults to "same".
        use_bias (bool, optional): Pass to layers.Conv2D. Defaults to False.

    Returns:
        Tensor: Tensor of one one convulutional layer (tf.layers.Conv2D)
    """
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    """Combines outputs of neurons to reduce resolution and focus on different sets of features

    Args:
        dspp_input (Tensor): Input for the pyramid

    Returns:
        Tensor: _description_
    """
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus_m1(image_size, num_classes):
    """Model itself

    Args:
        image_size (int): square resolution of image
        num_classes (int): how many classe are we detecting

    Returns:
        keras.Model: Model of DeepLabV3+ 
    """
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

def read_image(image_path, mask=False):
    """Function to read image or image mask and parse it to tf.Tensor

    Args:
        image_path (string): Path to image
        mask (bool, optional): Is image mask ? => read in RGB or B&W mode. Defaults to False.

    Returns:
        Tensor: Tensor containing image
    """
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_image(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image

def load_data(image_list, mask_list):
    """Function to load all images and masks

    Args:
        image_list (string[]): Path to every single image
        mask_list (string[]): Path to every single image mask

    Returns:
        Tensors: Tensor with images and tensor with all masks
    """
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    """Function to generate dataset from lists of files

    Args:
        image_list (string[]): Paths to all images
        mask_list (string[]): Paths to all image masks

    Returns:
        Tesnor: Dataset tensor
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

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
    model = g_unet(img_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    #model.summary() 
    #model = models.unet_3plus_2d((IMAGE_SIZE, IMAGE_SIZE,3), NUM_CLASSES, filter_num_down=[64, 128, 256, 512, 1024], filter_num_skip=[64, 64, 64, 64], stack_num_down=10, stack_num_up=10, backbone='ResNet50' )
    # Print model summary
    model.summary()

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
    
    outputFolder = './final_output'
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
