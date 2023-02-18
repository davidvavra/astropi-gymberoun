import logging
import threading

from modules.kthread import KThread
from modules.run_ai import AI
from modules import coverage
from modules import files

logger_thread = logging.getLogger("astropi.thread")
logger_main = logging.getLogger("astropi.main")


def start(base_folder, image, model="modules/model/q_PAN_MNV2-1024_INT8_edgetpu.tflite"):
    """Function that starts image classification in new thread and kills previous thread if it exists -> it likely got stuck

    Args:
        base_folder (str): base folder for all data
        image (str): path to image to process
        model (str, optional): path to model if different wanted. Defaults to "models/deeplab.tflite".
    """
    # Active threads count
    act_count = threading.active_count()
    # List of all currently running threads
    thr_list = threading.enumerate()
    logger_main.debug(f"Currently active threads: {act_count} <file: classification.py, fn: start>")
    # If more than 1 thread are running at the time (1 is main thread)
    if (act_count > 1):
        # Kill another thread
        logger_main.warning(f"There were too many threads so one will be killed <file: classification.py, fn: start>")
        try:
            thr_list[1].kill()
        except Exception as E:
            logger_main.critical(f'Unable to kill thread due to Exception: {E} <file: classification.py, fn: start>')
    # Start processing new image
    logger_main.info(f"Starting image processing on another thread with model {model} on image {image} <file: classification.py, fn:start>")
    try:
        t1 = KThread(target=_run_classification_in_thread, args=(base_folder, image, model))
        t1.start()
        logger_main.debug("Started another thread <file: classification.py, fn: start>")
    except Exception as E:
        logger_main.critical(f'Unable to start another thread due to Exception: {E} <file: classification.py, fn: start>')


def _run_classification_in_thread(base_folder, image, model):
    """Function that calls AI model on image and then processes outputed (calculates coverage)

    Args:
        base_folder (str): base folder for all data
        image (str): path to image to process
        model (str): path to model to use
    """
    # Define model
    logger_thread.debug(f"Initializing model on EdgeTPU <file: classification.py, fn: _run_classification_in_thread>")
    AI_model = AI(model, folder="images/masked")
    # Get path where final image is saved from AI model
    logger_thread.info(f"Processing image {image} on EdgeTPU <file: classification.py, fn: _run_classification_in_thread>")
    img_path = AI_model.run_model(image)
    logger_thread.info(f"Image {image} processed successfully and final mask was saved to {img_path}")
    try:
        coverage_data = coverage.get(img_path)
        logger_thread.info(f"Calculated coverage on image is: {coverage_data}")
        try:
            files.add_classification_csv_row(base_folder, img_path, coverage_data)
            logger_thread.debug(f"Coverage saved to CSV")
        except Exception as e:
            logger_thread.error(f'Failed save coverage to CSV: {e}')
    except Exception as e:
        logger_thread.error(f'Failed to calculate coverage from {img_path}: {e}')
