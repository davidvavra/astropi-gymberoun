from modules.kthread import KThread
from modules.run_ai import AI
import os
import threading
from time import sleep

def call_AI(image,model):
    """Function that calls AI model on image and then processes outputed (calculates coverage)

    Args:
        image (str): path to image to process
        model (str): path to model to use
    """
    # Define model
    AI_model = AI(model, folder="masked")
    # Get path where final image is saved from AI model
    img_path = AI_model.run_model(image)
    """
    TODO: Filipova funkce
    """

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
    # If more than 1 thread are running at the time (1 is main thread)
    if(act_count > 1):
        # Kill another thread
        thr_list[1].kill()
    # Start processing new image
    t1 = KThread(target=call_AI, args=(image,model,))
    t1.start()

def clean_thread():
    """Function to kill second thread
    """
    # Active threads count
    act_count = threading.active_count()
    # List of all currently running threads
    thr_list = threading.enumerate()
    # If more than 1 thread are running at the time (1 is main thread)
    if(act_count > 1):
        # Kill another thread
        thr_list[1].kill()