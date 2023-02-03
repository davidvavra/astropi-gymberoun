from pathlib import Path
import logging
from sense_hat import SenseHat
from time import sleep
from datetime import datetime, timedelta
import csv
import os
from PIL import Image

from picamera import PiCamera

#Importing our python programs
import modules
from modules.mask import m_process_image
from modules.ai_thread import start_classification
from modules.getlocation import get_location
from modules.get_sensor_data import get_sensor_data
from modules.save_csv import save_csv
from modules.create_folders import create_folder

camera = PiCamera()
resolution = (1296, 972)

camera.resolution = resolution


# base_folder = Path(__file__).parent.resolve()

base_folder = os.getcwd()


images_folder = "images/"

cropped_folder = "images/cropped/"

masked_folder = "images/masked/"

raw_image_folder = "images/raw/"


# Function for creating all folders
create_folder(base_folder)

logger = logging.getLogger("astropi")
#Creating logfile
try:
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(base_folder,"main.log"))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
except:
    print("Couldn't create logfile")

# Initialise the photo counter
counter = 1
# Record the start and current time
start_time = datetime.now()
now_time = datetime.now()



# Run a loop for (almost) three hours
while (now_time < start_time + timedelta(minutes=179)):

    # Taking the image
    photo = None
    try:
        photo = camera.capture(base_folder + "images/last_image.jpg")
        
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}')
    
    filename2 = None

    # Cropping the image
    try:
        
        outputimage = m_process_image(os.path.join(base_folder, photo))
        if outputimage: #if the image is unusable, m_process_image returns False
            logger.debug(f'Image considered usable')
            outputimage = outputimage[0]
            filename2 = cropped_folder + "image_croppped" + counter +  ".jpg"
            outputimage.save(filename2)
            
    
            try:    #giving the image to the AI if it is usable
                start_classification(filename2)
                logger.debug(f'main.py gave photo to the AI')
            except Exception as e:
                logger.error(f'{e.__class__.__name__}: {e} -- AI did not get the image')
        
        else:
            # If the image is considered unusable we do nothing
            logger.debug(f'Image considered unusable')
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e} -- Could not process')


    location = None

    try:
        location = get_location
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}')
    
    sensor_data = None

    try: 
        sensor_data = get_sensor_data
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}')


    try:
        save_csv(location, sensor_data, filename2, base_folder)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}')
        

    logger.info(f'end of loop number {counter}')
    counter += 1
    sleep(30)