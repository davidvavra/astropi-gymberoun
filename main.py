import os
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

from picamera import PiCamera
from sense_hat import SenseHat

# Import our own modules
from modules import logs
from modules import files
from modules import images
from modules import classification
from modules import iss_location
from modules import sensors

# Library setup
camera = PiCamera()
camera.resolution = (1820, 1024)
sense = SenseHat()

# File & folder setup
base_folder = Path(__file__).parent.resolve()
logger = logs.create_logger(base_folder)
files.create_folders(logger, base_folder)
files.create_csv_files(base_folder)

# Initialise the photo counter
counter = 0

# Record the start and current time
start_time = datetime.now()
now_time = datetime.now()

# Run a loop for (almost) three hours
while now_time < start_time + timedelta(minutes=178):
    # Start iteration
    counter += 1
    logger.info(f'Iteration {counter}')

    # Capture image
    image_file = None
    try:
        camera.capture(f"{base_folder}/{files.LAST_IMAGE_FILE}")
        # Crop image & prepare for classification
        image_file = images.process_image(base_folder, logger, counter)
        if image_file is not None:
            # Image considered usable for classification
            try:
                # Start classification in separate thread
                classification.start(logger, base_folder, image_file)
                logger.debug(f'Image passed to the classification thread')
            except Exception as e:
                logger.error(f'Classification failed: {e}')
        else:
            # Image not considered usable for classification
            logger.debug(f'Image considered unusable')
    except Exception as e:
        logger.error(f'Failed to take image: {e}')

    # Get ISS location
    location = None
    try:
        location = iss_location.get()
    except Exception as e:
        logger.error(f'Failed to get ISS location: {e}')

    # Get sensor data
    sensor_data = None
    try:
        sensor_data = sensors.get_data(sense)
    except Exception as e:
        logger.error(f'Failed to get sensor data: {e}')

    # Save everything to CSV
    try:
        files.add_data_csv_row(base_folder, location, sensor_data, image_file)
    except Exception as e:
        logger.error(f'Failed to save data to CSV: {e}')

    # Wait till next iteration
    sleep(30)
    now_time = datetime.now()
