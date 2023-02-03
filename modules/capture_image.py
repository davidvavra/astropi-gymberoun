from picamera import PiCamera
import os

def capture_image(base_folder):
    """
    Captures images
    """
    image = os.path.join(base_folder, "images/last_image.jpg")
    camera = PiCamera()
    camera.resolution = (1296, 972)
    # Capture the image
    camera.capture(image)
    
    return("images/last_image.jpg")
