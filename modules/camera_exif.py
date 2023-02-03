from time import sleep
from picamera import PiCamera
from pathlib import Path

def convert(angle):
    """
    překonvertuje úhel na .exif formát
    """
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle


base_folder = Path(__file__).parent.resolve()
camera = PiCamera()
camera.resolution = (1920, 1080)

sample_num = 5
sample_time = 10


sample_time -= 3
camera.start_preview(alpha = 200)
for i in range(sample_num):
    sleep(3)
    
    
    
    point = ISS.coordinates()

    
    # Convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(point.latitude)
    west, exif_longitude = convert(point.longitude)

    # Set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"
    
    
    
    camera.capture(f"{base_folder}/image%s.jpg" % i)
    print('Image captured')
    sleep(sample_time)
camera.stop_preview()