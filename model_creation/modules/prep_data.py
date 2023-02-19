# Import libraries
import os  # OS
from concurrent.futures import ThreadPoolExecutor

import cv2  # OpenCV2
import numpy as np  # Numpy
from PIL import Image


def prepare_data(data, mask_recolor_map=None):
    """Function that prepares data for AI (changes colors on mask images)
    Args:
        folders (string[]): folders with masks
        mask_recolor_map (np.array): how to recolor masks
    """
    file_name = data[0]
    folder = data[1]
    file = os.path.join(folder, file_name)

    # Load the input image
    image = cv2.imread(file)
    # Convert to HSV color mode for replacing colors
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # Use default if None was provided
    if(mask_recolor_map == None):
        # Colors of things in images to replace
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
    else:
        items = mask_recolor_map

    # Go thru all colros
    for item in items:
        # Create mask where that color is
        mask=cv2.inRange(hsv,item[0],item[1])
        # Replace color
        image[mask>0]=item[2]

    # Remove original .jpg file
    os.remove(file)
    # Save new file as .png
    cv2.imwrite(file.split(".")[0] + ".png", image)

def rotate_image(image, angle):
    """Funtcion for rotating image by n degrees
    Args:
        image (ncv2.Image): Image to rotate
        angle (int): Angle
    Returns:
        cv2.Image: Rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def create_rotation(data):
    """Function to create rotations for image. Creates copies of image each rotate by 2.5 degrees more than previous one.
    Args:
        data (array): contains file name and folder where image is
    """
    file_name = data[0]
    folder = data[1]
    file = os.path.join(folder, file_name)

    image = cv2.imread(file)

    for i in range(144):
        angle = (i+1)*2.5
        rotated_image = rotate_image(image, angle)
        cv2.imwrite(os.path.join(folder, file.split(".")[0] + f"_rot{angle}." + file.split(".")[1]), rotated_image)
        pass

def main(images, masks, mask_recolor_map=None, max_threads=15, rotate=True):
    """Function to process images to make them all-models compatible dataset
    Note: filenames for images and masks must be same (e.g. image_001.jpg and image_001.jpg but in different folder)

    Args:
        images (array): array containing paths to all folders containing images e.g.["/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/validation/images/", "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/train/images/"] -> will process all images in those two folders (any image type)
        masks (array): array containing paths to all folders containing masks e.g.["/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/validation/masks/", "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/train/masks/"] -> will process all masks in those two folders (any image type)
        mask_recolor_map (np.array, optional): array of colors that corresponds to categories in masked images. Defaults to None (will use built-it, colormap in the project README).
        max_threads (int, optional): how many concurrent threads for image processing can be run. Defaults to 15.
        rotate (boolean): Expand dataset by rotating each image and mask by 2.5° degrees. Defaults to True.
    """
    # Process masks (recolors to B&W)
    values = []
    for folder in masks:
        for file in os.listdir(folder):
            values.append([file, folder, mask_recolor_map])

    with ThreadPoolExecutor(max_threads) as exe:
                result = exe.map(prepare_data, values)

    if(rotate):
        # Process images and masks (rotate)
        values = []
        for folder in masks+images:
            for file in os.listdir(folder):
                values.append([file, folder])

        with ThreadPoolExecutor(max_threads) as exe:
                    result = exe.map(create_rotation, values)