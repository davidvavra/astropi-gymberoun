import argparse
import os

import cv2
import matplotlib as plt
import numpy as np
import tensorflow as tf
from PIL import Image


def read_image(image_path, mask=False):
    global image_size
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
        image = tf.image.resize(images=image, size=[image_size, image_size])
    else:
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[image_size, image_size])
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
    image = read_image(image_list, image_size=image_size)
    mask = read_image(mask_list, mask=True,image_size=image_size)
    return image, mask


def data_generator(image_list, mask_list,batch_size=1):
    """Function to generate dataset from lists of files
    Args:
        image_list (string[]): Paths to all images
        mask_list (string[]): Paths to all image masks
        batch_size (integer): Batch size argument for batching dataset (ds.batch()). Defaults to 1.
        image_size (integer): pass through
    Returns:
        Tensor: Dataset tensor
    """
    global image_size
    image_size = image_size
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def infer(model, image_tensor):
    """Function to run inference on TF model

    Args:
        model (tf.keras.Model): built and compiled tf.keras.Model to run inference on
        image_tensor (_type_): image tensor as input

    Returns:
        np.array: predicted mask
    """
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    """Functio to decode/encode colormap to image

    Args:
        mask (np.array): array containing B&W image from model inferecnce
        colormap (np.array): array containing colormap
        n_classes (number): how many classes at are in the colormap

    Returns:
        array: array (image_size, image_size, 3) RGB image from the original mask overlayed with colormap
    """
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
    """Create partialy seethrough overlay over original image

    Args:
        image (array): image
        colored_mask (array): overaly to put over image

    Returns:
        array: processed image with overlay (35% see through)
    """
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 4)):
    """Function just to show results

    Args:
        display_list (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (5, 4).
    """
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

def plot_predictions(images_list, colormap, model):
    """Wrapper arround all the model inferencing stuff

    Args:
        images_list (array): array of image to run inference on
        colormap (array): array containing colormap
        model (tf.keras.Model): built and compiled tf.keras.Model to run inference on
    """
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
    """Function to load image

    Args:
        name (str): path to image to load

    Returns:
        np.array: loaded Image converted to array
    """
    image= Image.open(name)
    return np.array(image)

def str2bool(v):
    """Function to converting sting to boolean

    Args:
        v (str): string to convert

    Raises:
        argparse.ArgumentTypeError: If unable to convert string to bool

    Returns:
        bool: value of string "in bool" 
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_colormap(path):
    """Function to load colormap from file

    Args:
        path (str): path to file containing colormap. If None or doesn't exist will return default colormap

    Raises:
        Exception: If colormap file has some issue

    Returns:
       np.array : colormap
    """
    if(not os.path.isfile(path)):
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
        return colormap
    f = open(path, 'r')
    lines = f.readlines()

    colormap = np.array()
    for line in lines:
        arr = line.split(",")
        if(len(arr) ==3):
            np.append(colormap,arr)
        else:
            raise Exception(f"Line: {line} doesn't fit standard.")
    return colormap

def get_recolor_map(path):
    """Load conversion recoloring map from file

    Args:
        path (str): file with recoloring maps

    Returns:
        _type_: _description_
    """
    if(not os.path.isfile(path)):
        unknown = [np.array([0,0,0]), np.array([20,255,255]), (5,5,5)]  # Jiné pevnina
        desert = [np.array([20,0,0]), np.array([50,255,255]), (1,1,1)]  # Poušť
        forest = [np.array([50,0,0]), np.array([60,255,255]), (2,2,2)]  # Les
        fields = [np.array([59,0,0]), np.array([75,255,255]), (3,3,3)]  # Louky
        river = [np.array([75,0,0]), np.array([105,255,255]), (4,4,4)]  # Řeka
        nothing = [np.array([105,0,0]), np.array([118,255,255]), (0,0,0)]   # Neurčito
        # !! otáčení obrázků => otočený prostor [0,0,0] => pletlo by se s neurčito/jiné (pevnina)
        ocean = [np.array([118,0,0]),np.array([130,255,255]), (6,6,6)]  # Oceán
        moutains = [np.array([130,0,0]), np.array([140,255,255]), (7,7,7)]  # Hory
        clouds = [np.array([140,0,0]), np.array([170,255,255]), (8,8,8)]    # Mraky
        islands = [np.array([170,0,0]), np.array([180,255,255]), (9,9,9)]   # Ostrovy

        over = [np.array([180,0,0]), np.array([255,255,255]), (0,0,0)]  # Neurčito

        # All arays
        items = [nothing, clouds, ocean, moutains, forest, river, unknown, islands, fields, desert, over]
        return items
    
    f = open(path, 'r')
    lines = f.readlines()

    colormap = np.array()
    for line in lines:
        arr = line.split(";")
        item = np.array(np.array(arr[0].split(",")), np.array(arr[1].split(",")), (arr[2],arr[2],arr[2]))
        np.append(colormap, item)
    return colormap