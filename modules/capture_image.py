from picamera import PiCamera

def capture_image(base_folder):
    camera = PiCamera()
    camera.resolution = (1296, 972)
    # Capture the image
    camera.capture(base_folder + "/images/last_image.jpg")
    
    return("images/last_image.jpg")