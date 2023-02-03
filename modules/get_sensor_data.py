from sense_hat import SenseHat
from datetime import datetime

def get_sensor_data():
    """
    Collects data form sensors
    """
    # Set up Sense Hat
    sense = SenseHat()
    sense_data = []

    # Get environmental data
    try:
        sense_data.append(sense.get_temperature())
    except:
        sense_data.append(-6969)
    try:
        sense_data.append(sense.get_pressure())
    except:
        sense_data.append(-6969)
    try:
        sense_data.append(sense.get_humidity())
    except:
        sense_data.append(-6969)

    # Get orientation data
    orientation = sense.get_orientation()
    try:
        sense_data.append(orientation["yaw"])
    except:
        sense_data.append(-6969)
    try:
        sense_data.append(orientation["pitch"])
    except:
        sense_data.append(-6969)
    try:
        sense_data.append(orientation["roll"])
    except:
        sense_data.append(-6969)

    # Get compass data
    mag = sense.get_compass_raw()
    try:
        sense_data.append(mag["x"])
    except:
        sense_data.append(-6969)   
    try:
        sense_data.append(mag["y"])
    except:
        sense_data.append(-6969)   
    try:
        sense_data.append(mag["z"])
    except:
        sense_data.append(-6969)  

    # Get accelerometer data
    acc = sense.get_accelerometer_raw()
    try:
        sense_data.append(acc["x"])
    except:
        sense_data.append(-6969)  
    try:
        sense_data.append(acc["y"])
    except:
        sense_data.append(-6969)  
    try:
        sense_data.append(acc["z"])
    except:
        sense_data.append(-6969)

    # Get gyroscope data
    gyro = sense.get_gyroscope_raw()
    try:
        sense_data.append(gyro["x"])
    except:
        sense_data.append(-6969)
    try:
        sense_data.append(gyro["y"])
    except:
        sense_data.append(-6969)
    try:
        sense_data.append(gyro["z"])
    except:
        sense_data.append(-6969)

    # Get the date and time
    sense_data.append(datetime.now())

    return sense_data


