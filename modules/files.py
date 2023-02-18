import csv
import os
from datetime import datetime

from urllib3.filepost import writer

# Folder names
IMAGES_FOLDER = "images"
RAW_IMAGES_FOLDER = f"{IMAGES_FOLDER}/raw"
CROPPED_IMAGES_FOLDER = f"{IMAGES_FOLDER}/cropped"
MASKED_IMAGES_FOLDER = f"{IMAGES_FOLDER}/masked"
CSV_FOLDER = "csv"
LOGS_FOLDER = "logs"

# File names
LAST_IMAGE_FILE = f"{IMAGES_FOLDER}/last_image.jpg"
DATA_CSV_FILE = f"{CSV_FOLDER}/data.csv"
CLASSIFICATION_CSV_FILE = f"{CSV_FOLDER}/classification.csv"


def create_folders(logger, base_folder):
    # Create images folder
    try:
        if not (os.path.isdir(base_folder / IMAGES_FOLDER)):
            os.mkdir(base_folder / IMAGES_FOLDER)
            logger.info(f"Created directory 'images'")
        else:
            logger.info(f"Directory 'images' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'images' directory")

    # Create images/raw folder
    try:
        if not (os.path.isdir(base_folder / RAW_IMAGES_FOLDER)):
            os.mkdir(base_folder / RAW_IMAGES_FOLDER)
            logger.info(f"Created directory 'images/raw'")
        else:
            logger.info(f"Directory 'images/raw' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'images/raw' directory")

    # Create images/cropped folder
    try:
        if not (os.path.isdir(base_folder / CROPPED_IMAGES_FOLDER)):
            os.mkdir(base_folder / CROPPED_IMAGES_FOLDER)
            logger.info(f"Created directory 'images/cropped'")
        else:
            logger.info(f"Directory 'images/cropped' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'images/cropped' directory")

    # Create images/masked folder
    try:
        if not (os.path.isdir(base_folder / MASKED_IMAGES_FOLDER)):
            os.mkdir(base_folder / MASKED_IMAGES_FOLDER)
            logger.info(f"Created directory 'images/masked'")
        else:
            logger.info(f"Directory 'images/masked' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'images/masked' directory")

    # Create csv folder
    try:
        if not (os.path.isdir(base_folder / CSV_FOLDER)):
            os.mkdir(base_folder / CSV_FOLDER)
            logger.info(f"Created directory 'csv'")
        else:
            logger.info(f"Directory 'csv' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'csv' directory")


def create_csv_files(base_folder):
    with open(base_folder / DATA_CSV_FILE, 'w') as f:
        writer = csv.writer(f)
        header = (
            "Date/time", "Image", "Temperature", "Pressure", "Humidity", "Yaw", "Pitch", "Row", "Mag_x", "Mag_y",
            "Mag_z", "Acc_x",
            "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z", "Latitude", "Longitude")
        writer.writerow(header)

    with open(base_folder / CLASSIFICATION_CSV_FILE, 'w') as f:
        writer = csv.writer(f)
        header = ("Image", "Ocean", "River", "Clouds", "Forest", "Field", "Desert", "Unknown land", "Unknown", "Island",
                  "Mountains")
        writer.writerow(header)


def add_data_csv_row(base_folder, location, sensor_data, image_path):
    csv_data = []
    # Add date and time
    csv_data.append(datetime.now())
    # Add image path
    if image_path is not None:
        csv_data.append(image_path)
    else:
        csv_data.append("")
    # Add sensor data
    if sensor_data is not None:
        csv_data += sensor_data
    else:
        csv_data += ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
    # Add ISS location data
    if location is not None:
        csv_data += location
    else:
        csv_data += ["", ""]
    # Write CSV row
    with open(base_folder / DATA_CSV_FILE, 'a', buffering=1, newline='') as f:
        data_writer = writer(f)
        data_writer.writerow(csv_data)


def add_classification_csv_row(base_folder, image_path, coverage):
    csv_data = []
    # Add date and time
    csv_data.append(datetime.now())
    # Add image path
    if image_path is not None:
        csv_data.append(image_path)
    else:
        csv_data.append("")
    # Add coverage data
    csv_data += coverage
    # Write CSV row
    with open(base_folder / CLASSIFICATION_CSV_FILE, 'a', buffering=1, newline='') as f:
        data_writer = csv.writer(f)
        data_writer.writerow(csv_data)
