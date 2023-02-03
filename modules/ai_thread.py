from modules.kthread import KThread
from modules.run_ai import AI
import os
import threading
from modules.save_coverage import save_coverage
from time import sleep
import logging
logger = logging.getLogger("astropi.ai_thread")

def call_AI(image,model):
    """Function that calls AI model on image and then processes outputed (calculates coverage)

    Args:
        image (str): path to image to process
        model (str): path to model to use
    """
    # Define model
    logger.debug(f"Initializing model on EdgeTPU")
    AI_model = AI(model, folder="masked")
    # Get path where final image is saved from AI model
    logger.info(f"Processing image {image} on EdgeTPU")
    img_path = AI_model.run_model(image)
    logger.info(f"Image {image} processed successfully and final mask was saved to {img_path}")
    logger.info(f"Calculated coverage on image is: {save_coverage(img_path)}")

def start_classification(image, model="models/deeplab.tflite"):
    """Function that starts image classification in new thread and kills previous thread if it exists -> it likely got stuck

    Args:
        image (str): path to image to process
        model (str, optional): path to model if different wanted. Defaults to "models/deeplab.tflite".
    """
    # Active threads count
    act_count = threading.active_count()
    # List of all currently running threads
    thr_list = threading.enumerate()
    logger.debug(f"Currently active threads: {act_count}")
    # If more than 1 thread are running at the time (1 is main thread)
    if(act_count > 1):
        # Kill another thread
        thr_list[1].kill()
        logger.debug(f"There were too many threads so one was killed")
    # Start processing new image
    logger.info(f"Starting image processing on another thread with model {model} on image {image}")
    t1 = KThread(target=call_AI, args=(image,model,))
    t1.start()
    logger.debug("Started another thread")

def clean_thread():
    """Function to kill second thread
    """
    # Active threads count
    act_count = threading.active_count()
    logger.debug(f"Currently active threads: {act_count}")
    # List of all currently running threads
    thr_list = threading.enumerate()
    # If more than 1 thread are running at the time (1 is main thread)
    if(act_count > 1):
        # Kill another thread
        thr_list[1].kill()
        logger.info("Killed all non-main threads")