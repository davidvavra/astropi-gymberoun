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

#Function for creating all folders
create_folder(base_folder)


# Initialise the photo counter
counter = 1
# Record the start and current time
start_time = datetime.now()
now_time = datetime.now()



# Run a loop for (almost) three hours
while (now_time < start_time + timedelta(minutes=179)):

    #taking the image
    photo = None
    try:
        photo = capture_image(base_folder)
        
    except Exception as err:
        print(f"Capturing failed because of  {err}")
    
    #cropping the image
    try:

        outputimage = m_process_image(os.path.join(base_folder, photo))
        if outputimage: #if the image is unusable, m_process_image returns False
            outputimage = outputimage[0]
            filename2 = cropped_folder + "image_croppped" + counter +  ".jpg"
            outputimage.save(filename2)

            try:    #giving the image to the AI if it is usable
                start_classification(filename2)
            except:
                print("Couldnt find function")
        #if the image is considered unusable we do nothing
        else:
            filename2 = "unusable"
    except:
        photo.save(raw_image_folder + "raw_image" + counter + ".jpg")

    
    try:
        location = get_location
    except:
        print(":(")
    
    try: 
        sensor_data = get_sensor_data
    except:
        print("couldnt get sensor data")

    try:
        save_csv(filename2, sensor_data, location)
    except:
        print("couldnt save location")
        
    counter += 1
    sleep(30)