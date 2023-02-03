from time import sleep
from picamera import PiCamera
from pathlib import Path

base_folder = Path(__file__).parent.resolve()
camera = PiCamera()
camera.resolution = (1920, 1080)

sample_num = 5
sample_time = 10

sample_time -= 3
camera.start_preview(alpha = 200)
for i in range(sample_num):
    sleep(3)
    camera.capture(f"{base_folder}/image%s.jpg" % i)
    print('Image captured')
    sleep(sample_time)
camera.stop_preview()