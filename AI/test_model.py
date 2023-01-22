# Import libraries
import datetime
import os
from scipy.io import loadmat
from glob import glob
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorboard
import tensorflow as tf

# Matplotlib settings
mpl.use('TkAgg')

IMAGE_SIZE = 800
DATA_DIR = "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset"      # Where are my data ? XD
MODEL = 'output/GU-Net_v1_noDropout-800px-814p.hdf5'

model = tf.keras.models.load_model(MODEL)
model.summary()

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

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


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
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
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 10)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(8, 4)
        )

if __name__ == "__main__":
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
    plot_predictions([os.path.join(DATA_DIR, "train backup/images/matyas_008.jpg")], colormap, model=model)