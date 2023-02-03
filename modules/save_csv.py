from csv import writer
from importlib import import_module
import os

def save_csv(location, sensor_data, crop_image, base_folder):
    """
    Saves data to CSV file
    """

    csv_data = []
    data_file = os.path.join(base_folder, "csv/data.csv")
    try:
        csv_data.append(crop_image)
    except:
        csv_data.append("placeholder")
    try:
        csv_data += sensor_data
    except:
        csv_data += ["","","","","","","","","","","","","","","",""]
    try:
        csv_data += location
    except:
        csv_data += ["",""]

    with open(data_file, 'a', buffering=1, newline='') as f:
        data_writer = writer(f)
        data_writer.writerow(csv_data)
