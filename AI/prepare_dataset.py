# Import libraries
from concurrent.futures import ThreadPoolExecutor
import os  # OS

import cv2  # OpenCV2
import numpy as np  # Numpy

def prLightPurple(skk): print("\033[94m{}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))

def prepare_data(data, output=False):
    """Function that prepares data for AI (changes colors on mask images)

    Args:
        folders (string[]): folders with masks
        output (bool): print to terminal
    """
    file_name = data[0]
    folder = data[1]
    #if(output):
    #    prLightPurple(f"----------\nPreparing data\n----------")
    # For each folder
    #for folder in folders:
     #   if(output):
     #       prCyan(f"*** Processing folder {folder} ***")
      #  # For every file
      #  for file in os.listdir(folder):
    # File path (os.listdir) return only filenames, eg. image1_mask.jpg
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
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def create_rotation(data, output=False):
    #if(output):
    #    prLightPurple(f"----------\nRotating images\n----------")
    #for folder in folders:
    #    if(output):
    #        prCyan(f"*** Processing folder {folder} ***")
    #    for file in os.listdir(folder):
    file_name = data[0]
    folder = data[1]
    file = os.path.join(folder, file_name)
    
    if(output):
        prGreen(f">> Processing file {file}")

    image = cv2.imread(file)

    for i in range(144):
        angle = (i+1)*2.5
        rotated_image = rotate_image(image, angle)
        cv2.imwrite(os.path.join(folder, file.split(".")[0] + f"_rot{angle}." + file.split(".")[1]), rotated_image)
        pass

if __name__ == "__main__":
    masks = ["/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/validation/masks/", "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/train/masks/"]
    images = ["/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/validation/images/", "/home/cyril/Documents/astropi-gymberoun/local/AI/final_dataset/train/images/"]
    
    values = []
    for folder in masks:
        for file in os.listdir(folder):
            values.append([file, folder])

    with ThreadPoolExecutor(15) as exe:
                result = exe.map(prepare_data, values)
    print(result)

    values = []
    for folder in masks+images:
        print(folder)
        for file in os.listdir(folder):
            values.append([file, folder])
    print(result)

    with ThreadPoolExecutor(15) as exe:
                result = exe.map(create_rotation, values)


    #prepare_data(folders = masks, output=True)
    #create_rotation(folders = masks+images, output=True)