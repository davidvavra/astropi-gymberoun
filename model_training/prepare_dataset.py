# Import libraries
import os  # OS
from concurrent.futures import ThreadPoolExecutor, process

import cv2  # OpenCV2
import mask  # Another file import
import numpy as np  # Numpy

# Functions for printing in different colors to make output more human friendly
def prRed(skk): print("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))

def color_mask(data, output=True):
    """Function that prepares data for AI (changes colors on mask images)

    Args:
        folders (string[]): folders with masks
        output (bool): print to stdout ? Defaults to True.
    """
    file_name = data[0]
    folder = data[1]
    file = os.path.join(folder, file_name)
    
    # Show that we are working XD
    if(output):
        prGreen(f">> Processing file: {file}")
    # Load the input image
    image = cv2.imread(file)
    # Convert to HSV color mode for replacing colors
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

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
    """Function to rotate image by given angle

    Args:
        image (cv2.Image): Image to rotate
        angle (int): Angle

    Returns:
        cv2.Image: Rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def create_rotation(data, output=True):
    """Function to create 144 rotated copies of file (each rotated by 2.5° more than previous)

    Args:
        data (string[]): [file_name, folder]
        output (bool, optional): Print to stdout ?. Defaults to True.
    """
    file_name = data[0]
    folder = data[1]
    file = os.path.join(folder, file_name)
    
    # if output => print 
    if(output):
        prGreen(f">> Processing file {file}")

    # Load image
    image = cv2.imread(file)

    # Rotate image
    for i in range(144):
        # Calculate angle
        angle = (i+1)*2.5
        # Rotate image
        rotated_image = rotate_image(image, angle)
        # Save to file
        cv2.imwrite(os.path.join(folder, file.split(".")[0] + f"_rot{angle}." + file.split(".")[1]), rotated_image)

def process_image(data, output=True):
    file_name = data[0]
    folder = data[1]
    mask_name = data[2]
    mask_folder = data[3]
    file = os.path.join(folder, file_name)
    mask_file = os.path.join(mask_folder, mask_name)
    
    if(output):
        prGreen(f">> Processing file {file}")

    processed = mask.m_process_image(file, image_mask=mask_file)
    if(processed == False):
        prRed(f">> Deleting file {file} and its mask {mask_file} due to image being unusable")
        os.remove(file)
        os.remove(mask_file)
    else:
        processed[0].save(file)
        processed[1].save(mask_file)


if __name__ == "__main__":
    # Where to look for masks
    masks = ["/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/validation/masks/", "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/train/masks/"]
    # Where to look for iamges
    images = ["/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/validation/images/", "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/train/images/"]
    # How many concurrent threads to run
    num_threads = 20
    
    # Prepare values for (re)coloring masks
    values = []
    for folder in masks:
        for file in os.listdir(folder):
            # File to process and folder in which file is located
            values.append([file, folder])

    # Recolor masks
    prCyan("-----------------------------------------")
    prCyan(">>>>> Recoloring masks <<<<<")
    prCyan("-----------------------------------------")
    with ThreadPoolExecutor(num_threads) as exe:
                result = exe.map(color_mask, values)

    # Prepare values for rotating images and masks
    values = []
    for folder in masks+images:
        for file in os.listdir(folder):
            # File to process and folder in which file is located
            values.append([file, folder])

    # Rotating images and masks
    prCyan("-----------------------------------------")
    prCyan(">>>>> Rotating images and masks <<<<<")
    prCyan("-----------------------------------------")
    with ThreadPoolExecutor(num_threads) as exe:
                result = exe.map(create_rotation, values)
    
    # Prepare values for cropping and filtering images
    values = []
    for index,folder in enumerate(images):
        for index2,file in enumerate(os.listdir(folder)):
            # File to process and folder in which file is located
            mask_folder = masks[index]
            mask_file = os.listdir(masks[index])[index2]
            values.append([file, folder, mask_file, mask_folder])
        
    # Cropping and filtering images
    prCyan("-----------------------------------------")
    prCyan(">>>>> Cropping and filtering images <<<<<")
    prCyan("-----------------------------------------")
    with ThreadPoolExecutor(num_threads) as exe:
        result = exe.map(process_image, values)