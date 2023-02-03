from csv import writer
from importlib import import_module
import os

def save_csv(location, sensor_data, crop_image, base_folder):
    """
    Saves data to CSV file
    """

    csv_data = []
    data_file = os.path.join(base_folder, "csv/data.csv")
    if crop_image != None:
        csv_data.append(crop_image)
    else:
        csv_data.append("placeholder")
    if sensor_data != None:
        csv_data += sensor_data
    else:
        csv_data += ["","","","","","","","","","","","","","","",""]
    if csv_data != None:
        csv_data += location
    else:
        csv_data += ["",""]

    with open(data_file, 'a', buffering=1, newline='') as f:
        data_writer = writer(f)
        data_writer.writerow(csv_data)

