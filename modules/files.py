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
    _create_folder(logger, base_folder, IMAGES_FOLDER)
    _create_folder(logger, base_folder, RAW_IMAGES_FOLDER)
    _create_folder(logger, base_folder, CROPPED_IMAGES_FOLDER)
    _create_folder(logger, base_folder, MASKED_IMAGES_FOLDER)
    _create_folder(logger, base_folder, CSV_FOLDER)


def create_csv_files(base_folder):
    with open(f"{base_folder}/{DATA_CSV_FILE}", 'w') as f:
        writer1 = csv.writer(f)
        header = (
            "Date/time", "Image", "Temperature", "Pressure", "Humidity", "Yaw", "Pitch", "Row", "Mag_x", "Mag_y",
            "Mag_z", "Acc_x",
            "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z", "Latitude", "Longitude")
        writer1.writerow(header)

    with open(f"{base_folder}/{CLASSIFICATION_CSV_FILE}", 'w') as f:
        writer2 = csv.writer(f)
        header = ("Image", "Ocean", "River", "Clouds", "Forest", "Field", "Desert", "Unknown land", "Unknown", "Island",
                  "Mountains")
        writer2.writerow(header)


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
    with open(f"{base_folder}/{DATA_CSV_FILE}", 'a', buffering=1, newline='') as f:
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
    with open(f"{base_folder}/{CLASSIFICATION_CSV_FILE}", 'a', buffering=1, newline='') as f:
        data_writer = csv.writer(f)
        data_writer.writerow(csv_data)


def _create_folder(logger, base_folder, folder):
    try:
        path = f"{base_folder}/{folder}"
        if not (os.path.isdir(path)):
            os.mkdir(path)
            logger.info(f"Created directory '{folder}'")
        else:
            logger.info(f"Directory '{folder}' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating '{folder}' directory")
