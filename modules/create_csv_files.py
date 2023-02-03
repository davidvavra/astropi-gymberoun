import csv

def create_csv_file(base_folder):
    
    """
    Create a new CSV file and add the header row
    
    """
    
    data_file = base_folder + "/csv/data.csv"
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Image", "Temperature", "Preassure", "Humidity", "Yaw", "Pitch", "Row", "Mag_x", "Mag_y", "Mag_z", "Acc_x", "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z", "Date/time", "Latitude", "Longitude",)
        writer.writerow(header)

    masked_file = base_folder + "/csv/masked.csv"
    with open(masked_file, 'w') as g:
        writer1 = csv.writer(g)
        header1 = ("Image", "Ocean", "River", "Clouds", "Forest", "Field", "Desert", "Unkwonland", "Unknown", "Island", "Mountains")
        writer1.writerow(header1)



    


