from pathlib import Path
from logzero import logger, logfile
from sense_hat import SenseHat
from time import sleep
from datetime import datetime, timedelta
import csv
import os
from PIL import Image


import mask             #importing our python programs
from csv_writing import camera_exif
 
base_folder = Path(__file__).parent.resolve()

images_folder = "images/"

cropped_folder = "images/cropped/"

masked_folder = "images/masked/"

raw_image_folder = "images/raw/"

filename2 = "unusable"




# Initialise the photo counter
counter = 1
# Record the start and current time
start_time = datetime.now()
now_time = datetime.now()

# Run a loop for (almost) three hours
while (now_time < start_time + timedelta(minutes=179)):

    #taking the image
    try:
        photo = camera_exif.capture_image(base_folder)
        
    except:
        print("Capturing failed")
    
    #cropping the image
    try:

        outputimage = mask.m_process_image(os.path.join(base_folder, photo))
        if outputimage: #if the image is unusable, m_process_image returns False
            outputimage = outputimage[0]
            filename2 = cropped_folder + "image_croppped" + counter +  ".jpg"
            outputimage.save(filename2)

            try:    #giving the image to the AI if it is usable
                AI.start_classification(filename2)
            except:
                print("Couldnt find function")
        #if the image is considered unusable we do nothing
        else:
            filename2 = "unusable"
    except:
        photo.save(raw_image_folder + "raw_image" + counter + ".jpg")

    
    try:
        location = camera_exif.get_location
    except:
        print(":(")
    
    try:
        camera_exif.save_location(filename2, location_A)
    except:
        print("couldnt save location")

    try: 
        sensor_data = camera_exif.get_sensor_data
    except:
        print("couldnt get sensor data")

    try:
        camera_exif.save_csv(sensor_data)
    except:
        print("couldnt save sensor data")

    counter += 1
    sleep(30)