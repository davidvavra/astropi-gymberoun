import tensorflow as tf
import numpy as np
import os
from PIL import Image

from keras.models import load_model

IMAGE_SIZE = 800
NUM_CLASSES = 10

INPUT_IMAGES = "C:\\Users\\cyrda\\Documents\\Barrande\\images\\cropped"
PROCESSED_IMAGES = "C:\\Users\\cyrda\\Documents\\Barrande\\images\\my"

MODEL = "C:\\Users\\cyrda\\Downloads\\model.hdf5"

model = load_model(MODEL)

model.summary()


def read_image(image_path, mask=False, img_size=800):
    image_size = img_size
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

def plot_predictions(images_list, colormap, model, fname, img_size):
    """Wrapper arround all the model inferencing stuff

    Args:
        images_list (array): array of image to run inference on
        colormap (array): array containing colormap
        model (tf.keras.Model): built and compiled tf.keras.Model to run inference on
    """
    for image_file in images_list:
        image_tensor = read_image(image_file, img_size=img_size)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 10)
        #overlay = get_overlay(image_tensor, prediction_colormap)
        img = Image.fromarray(prediction_colormap)
        img.save(fname)

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

def main():
    for file in os.listdir(INPUT_IMAGES):
        fl = file.split(".")
        fl[0] = fl[0] + "_masked"
        file2 = ".".join(fl)
        print(file)
        print(os.path.join(INPUT_IMAGES,file))
        print(file2)
        print(os.path.join(INPUT_IMAGES, file2))
        print("-----------------------")
        plot_predictions([os.path.join(INPUT_IMAGES, file)], load_colormap("invalid path"), model=model, fname=os.path.join(PROCESSED_IMAGES, file2), img_size=IMAGE_SIZE)

if __name__ == "__main__":
    main()