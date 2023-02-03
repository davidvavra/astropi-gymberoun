from pathlib import Path
from csv import writer
from sense_hat import SenseHat
from datetime import datetime
import random

sense = SenseHat()
base_folder = Path(__file__).parent.resolve()
data_file = base_folder / "data.csv" #create a folder for csv data file

delay = 1
timestamp = datetime.now()

def get_sense_data():
    sense_data = []
    # Get environmental data
    sense_data.append(sense.get_temperature())
    sense_data.append(sense.get_pressure())
    sense_data.append(sense.get_humidity())
    #Get orientation data
    orientation = sense.get_orientation()
    sense_data.append(orientation["yaw"])
    sense_data.append(orientation["pitch"])
    sense_data.append(orientation["roll"])
    # Get compass data
    mag = sense.get_compass_raw()
    sense_data.append(mag["x"])
    sense_data.append(mag["y"])
    sense_data.append(mag["z"])
    # Get accelerometer data
    acc = sense.get_accelerometer_raw()
    sense_data.append(acc["x"])
    sense_data.append(acc["y"])
    sense_data.append(acc["z"])
    #Get gyroscope data
    gyro = sense.get_gyroscope_raw()
    sense_data.append(gyro["x"])
    sense_data.append(gyro["y"])
    sense_data.append(gyro["z"])
    #Get the date and time
    sense_data.append(datetime.now())

    return sense_data

with open(data_file, 'w', buffering=1, newline='') as f:
    data_writer = writer(f) 
    data_writer.writerow(['temp', 'pres', 'hum',
                          'yaw', 'pitch', 'roll',
                          'mag_x', 'mag_y', 'mag_z',
                          'acc_x', 'acc_y', 'acc_z',
                          'gyro_x', 'gyro_y', 'gyro_z',
                          'datetime'])

    while True: #writing data to file
        data = get_sense_data()
        time_difference = data[-1] - timestamp
        print(time_difference)
        if time_difference.seconds > delay:
            data_writer.writerow(data)
            timestamp = datetime.now()