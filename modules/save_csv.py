# from csv import writer


# def create_csv_file(data_file):
#     """Create a new CSV file and add the header row"""
#     with open(data_file, 'w') as f:
#         writer = csv.writer(f)
#         header = ("Image", "Temperature", "Preassure", "Humidity", "Yaw", "Pitch", "Row", "Mag_x","Mag_y","Mag_z","Acc_x","Acc_y","Acc_z","Gyro_x","Gyro_y","Gyro_z", "Date/time", "Latitude", "Longitude",)
#         writer.writerow(header)


# def add_csv_data(data_file, data):
#     """Add a row of data to the data_file CSV"""
#     with open(data_file, 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(data)


# # Initialise the CSV file
# data_file = base_folder/csv/"data.csv"
# create_csv_file(data_file)

# def Write():
#     with open(data_file, 'a', buffering=1, newline='') as f:
#         data_writer = writer(f)
#         data = get_sense_data()
#         data_writer.writerow(data)

def save_csv(location, sensor_data, crop_image):
    pass
