import cv2
import numpy as np
import logging
logger = logging.getLogger("astropi.thread")


def get(filename):
    """
    Get masked image coverage in text form.

    Args:
        filename: masked image
    """
    logger.debug(f'Calculating coverage of mask...')
    coverage = []

    # Colors
    ocean = [215, 0, 0]
    river = [215, 255, 0]
    clouds = [215, 181, 227]
    forest = [0, 89, 7]
    field = [0, 255, 7]
    desert = [0, 255, 255]
    unknownland = [0, 106, 255]
    unknown = [220, 106, 90]
    island = [129, 106, 255]
    mountains = [129, 0, 67]

    colors =[ocean, river, clouds, forest, field, desert, unknownland, unknown, island, mountains]

    # Read the images
    img = cv2.imread(filename)

    # Convert Image to Image HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in colors:
        # Get color
        color = np.uint8([[i]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # Defining lower and upper bound HSV values
        h = hsv_color[0][0][0]
        lower = np.array([h, 50, 50])
        upper = np.array([h, 255, 255])

        mask = cv2.inRange(hsv, lower, upper) 

        # Get average color
        array = np.array(mask)
        average_color = np.mean(array) / 2.55
        coverage.append(average_color)
    logger.info(f'Calculated mask coverage is {coverage} (ocean, river, clouds, field, deser, unknown_land, unknown, island, mountains)')
    return coverage

