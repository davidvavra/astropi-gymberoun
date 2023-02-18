from code import interact
from glob import glob
import os
import random
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from deeplab_v3_plus import DeepLabV3Plus
from unet import UNet
from fcn import FCN
from pan import PAN
from tensorflow.keras.applications import imagenet_utils

IMAGE_SIZE = 1024
NUM_CLASSES = 10
BATCH_SIZE = 1
DATA_DIR = "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset"
MODE = 3 # 1 - train, 2 - test, 3 - convert, 4 - evaluate, 5 - not working
MODEL = 'models/DLAB-VGG16-1024.hdf5'

mpl.use('TkAgg')

gpus = tf.config.experimental.list_physical_devices('GPU')
# Set memory growth to True 
# (will prevent some fails if low memory and improve memory management while slightly slowing progress) 
# - stability imporvement
tf.config.experimental.set_memory_growth(gpus[0], True)

# Create model
# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

def create_model(compile=False):
    #model= DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    #model= UNET(img_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    model= DeepLabV3Plus(NUM_CLASSES, version='DeepLabV3Plus', base_model="VGG16")(input_size=(IMAGE_SIZE, IMAGE_SIZE))
    #model = UNet(NUM_CLASSES, base_model='MobileNetV2')(input_size=(IMAGE_SIZE, IMAGE_SIZE))
    #model= FCN(NUM_CLASSES, base_model='MobileNetV2')(input_size=(IMAGE_SIZE, IMAGE_SIZE))
    #model = PAN(NUM_CLASSES, base_model="MobileNetV2")(input_size=(IMAGE_SIZE, IMAGE_SIZE))
    if(compile):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
    return model


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


def data_generator(image_list, mask_list, idk=True):
    """Function to generate dataset from lists of files
    Args:
        image_list (string[]): Paths to all images
        mask_list (string[]): Paths to all image masks
    Returns:
        Tesnor: Dataset tensor
    """
    if(idk):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        return dataset
    else:
        dataset = tf.data.Dataset.from_tensor_slices((image_list)).batch(1)
        return dataset

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = tf.keras.Input(shape=(image_size, image_size, 3))
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = tf.keras.layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = tf.keras.layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return tf.keras.Model(inputs=model_input, outputs=model_output)


def UNET(img_size, num_classes):
    img_size = (img_size,img_size)
    inputs = tf.keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = tf.keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model

#model.summary()

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    print(predictions.shape)
    print(predictions[0,400,400]);
    predictions = np.squeeze(predictions)
    print(predictions[400,400]);
    predictions = np.argmax(predictions, axis=2)
    print(predictions[400,400]);
    #print(f'Shape:{predictions.shape}')
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 4)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        print(f"image_tensor.shape: {image_tensor.shape}")
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        print(f'prediction_mask.shape : {prediction_mask.shape}')
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 10)
        print(f'prediction_colormap.shape : {prediction_colormap.shape}')
        print(prediction_colormap)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [prediction_mask,image_tensor, overlay, prediction_colormap], figsize=(8, 4)
        )

def load_image(name):
    image= Image.open(name)
    #print(img)
    #print('Load Image', name, np.array(img).shape)
    return np.array(image)

# TF_GPU_ALLOCATOR=cuda_malloc_async

if __name__ == "__main__" and MODE == 1:
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        f'models/checkpoint.hdf5', monitor='val_accuracy', verbose=1,
        save_best_only=True, save_weights_only=False,
        save_frequency=5)

    # List all training data (images & masks)
    train_images = sorted(glob(os.path.join(DATA_DIR, "train/images/*")))
    train_masks = sorted(glob(os.path.join(DATA_DIR, "train/masks/*")))

    # List all validation data (images & masks)
    val_images = sorted(glob(os.path.join(DATA_DIR, "validation/images/*")))
    val_masks = sorted(glob(os.path.join(DATA_DIR, "validation/masks/*")))

    # Creating training and validation datasets
    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    model.summary()
    model.fit(train_dataset, epochs=20, callbacks=[checkpoint_callback], validation_data=val_dataset, shuffle=True)
elif __name__ == "__main__" and MODE == 2:
    model = create_model()

    model.load_weights(MODEL)

    model.summary()
    train_images = sorted(glob(os.path.join(DATA_DIR, "train/images/*")))
    #print(train_images)

    colormap = np.array([[255,106,0],
                [255,255,0],
                [7,89,0],
                [7,255,0],
                [0,255,215],
                [90,106,220],
                [0,0,215],
                [67,0,129],
                [227,181,215],
                [255,106,129]])

    #plot_predictions(train_images[150:154], colormap, model=model)
    #print(os.path.join(DATA_DIR, "train_backup/images/cyril_027.jpg"))
    plot_predictions([os.path.join(DATA_DIR, "combined/images/combined_064.jpg")], colormap, model=model)
elif __name__ == "__main__" and MODE == 3:
    print(MODEL)
    # List all validation data (images & masks)
    images = sorted(glob(os.path.join(DATA_DIR, "validation/images/*")))
    masks = sorted(glob(os.path.join(DATA_DIR, "validation/masks/*")))

    # Creating training and validation datasets
    #dataset = data_generator(images, masks, idk=True)

    saved_tflite_model_path = "Saved_TFLite_Model"

    if not os.path.isdir(saved_tflite_model_path):
        os.makedirs(saved_tflite_model_path)

    filename=MODEL

    if os.path.isfile(filename):
        print('Model ', filename, ' exists')

        """model = tf.keras.models.load_model(filename, , custom_objects={
            'categorical_crossentropy_with_logits': categorical_crossentropy_with_logits})"""
        #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #model = DeepLabV3Plus(num_classes=10, base_model="MobileNetV2")(input_size=(1024,1024))
        model = create_model()

        model.load_weights(MODEL)

        model.summary()
        #print(model.get_layer("input_1").input_shape)

        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
        converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
        converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
        converter_float32_ = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
    
    images = glob(os.path.join(DATA_DIR, "validation/images/*"))
    val_images = sorted(glob(os.path.join(DATA_DIR, "validation/images/*")))
    val_masks = sorted(glob(os.path.join(DATA_DIR, "validation/masks/*")))

    # Creating training and validation datasets
    val_dataset = data_generator(val_images, val_masks)
    rep_ds = val_dataset.shuffle(128).take(100)
    def representative_data_gen():
        for image,mask in rep_ds:
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

    #test_accuracy = evaluate(interpreter)

elif __name__ == '__main__' and MODE == 4:
    model = create_model(compile=True)
    model.load_weights(MODEL)
    model.summary()
    print("Evaluate on combined data")
    images = sorted(glob(os.path.join(DATA_DIR, "combined/images/*")))
    masks = sorted(glob(os.path.join(DATA_DIR, "combined/masks/*")))    
    dataset = data_generator(images, masks)
    result = model.evaluate(dataset, batch_size=128)
    print("test loss, combined acc:", result)

elif __name__ == '__main__' and MODE == 5:
    interpreter = tf.lite.Interpreter("Saved_TFLite_Model/q_DLAB-VGG16-1024_INT8.tflite")
    print(interpreter)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)
    image = cv2.resize(load_image(os.path.join(DATA_DIR, "combined/images/combined_064.jpg")), dsize=(IMAGE_SIZE,IMAGE_SIZE))
    color_image = imagenet_utils.preprocess_input(image.astype(np.uint8), data_format='channels_last', mode='torch')
    color_image = np.expand_dims(color_image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], color_image.astype(np.uint8))
    print("Invoking")
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Output Data Shape", output_data.shape)
    if np.ndim(output_data) == 4:
                  prediction = np.squeeze(output_data, axis=0)
    prediction = Image.fromarray(np.uint8(prediction))

    plt.imshow(prediction)
    plt.show()
