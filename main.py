from pathlib import Path
from logzero import logger, logfile
from sense_hat import SenseHat
from time import sleep
from datetime import datetime, timedelta
import csv
import os
from PIL import Image

#Importing our python programs
import modules
from modules.mask import m_process_image
from modules.ai_thread import start_classification
from modules.capture_image import capture_image 
from modules.getlocation import get_location
from modules.get_sensor_data import get_sensor_data
from modules.save_csv import save_csv
from modules.create_folders import create_folder

 
base_folder = Path(__file__).parent.resolve()

images_folder = "images/"

cropped_folder = "images/cropped/"

masked_folder = "images/masked/"

raw_image_folder = "images/raw/"

filename2 = "unusable"

photo = None

#Function for creating all folders
create_folder(base_folder)

#Creating logfile
try:
    logfile(base_folder/"main.log")
except:
    print("Couldnt create logfile")

# Initialise the photo counter
counter = 1
# Record the start and current time
start_time = datetime.now()
now_time = datetime.now()



# Run a loop for (almost) three hours
while (now_time < start_time + timedelta(minutes=179)):

    #taking the image
    try:
        photo = capture_image(base_folder)
        
    except:
        logger.error(f'{e.__class__.__name__}: {e} -- Could not capture the image')
    
    #cropping the image
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
        #if the image is considered unusable we do nothing
        else:
            logger.debug(f'Image considered unusable')
            filename2 = "unusable"
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e} -- Could not process')
        try:
            photo.save(raw_image_folder + "raw_image" + counter + ".jpg")
        except:
            logger.error(f'{e.__class__.__name__}: {e} -- Could not save the raw image')

    
    try:
        location = get_location
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}')
    
    try: 
        sensor_data = get_sensor_data
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}')

    try:
        save_csv(filename2, sensor_data, location)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}')
        

    logger.info(f'end of loop number' + counter)
    counter += 1
    sleep(30)