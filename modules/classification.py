import logging
import threading

from modules.kthread import KThread
from modules.run_ai import AI
from modules import coverage
from modules import files

logger_main = logging.getLogger("astropi")
logger_thread = logging.getLogger("astropi.thread")


def clean_previous_classification_thread_if_needed():
    """
    Kills a previous thread if it exists. In that case it took more than 30s or got stuck.
    """
    # Active threads count
    act_count = threading.active_count()
    # List of all currently running threads
    thr_list = threading.enumerate()
    logger_main.debug(f"Currently active threads: {act_count}")
    # If more than 1 thread are running at the time (1 is main thread)
    if (act_count > 1):
        # Kill another thread
        logger_main.warning(f"There were too many threads so one will be killed")
        try:
            thr_list[1].kill()
        except Exception as E:
            logger_main.critical(f'Unable to kill thread due to Exception: {E}')


def start(base_folder, image, model="modules/model/q_PAN_MNV2-1024_INT8_edgetpu.tflite"):
    """Function that starts image classification in new thread.
    We use threading to cleanly separate machine learning classification, which takes longer to run and
    we don't want to affect the collecting of data on the main thread in any way.
    If something fails on the classification thread, we at least get the images & sensor data back.

    Args:
        base_folder (str): base folder for all data
        image (str): path to image to process
        model (str, optional): path to model if different wanted.
    """

    # Start processing new image
    logger_main.info(
        f"Starting image processing on another thread with model {model} on image {image}")
    try:
        t1 = KThread(target=_run_classification_in_thread, args=(base_folder, image, model))
        t1.start()
        logger_main.debug("Started another thread")
    except Exception as E:
        logger_main.critical(f'Unable to start another thread due to Exception: {E}')


def _run_classification_in_thread(base_folder, image, model):
    """Function that calls AI model on image and then processes outputed (calculates coverage)

    Args:
        base_folder (str): base folder for all data
        image (str): path to image to process
        model (str): path to model to use
    """

    logger_thread.debug(f"Initializing model on EdgeTPU")
    try:
        AI_model = AI(model, files.MASKED_IMAGES_FOLDER, base_folder)
        logger_thread.debug(
            f"Processing image {image} on EdgeTPU <fn: _run_classification_in_thread>")
        img_path = AI_model.run_model(image)
        try:
            coverage_data = coverage.get(img_path)
            try:
                files.add_classification_csv_row(base_folder, img_path, coverage_data)
                logger_thread.debug(f"Coverage saved to CSV")
            except Exception as e:
                logger_thread.error(f'Failed save coverage to CSV: {e}')
        except Exception as e:
            logger_thread.error(f'Failed to calculate coverage from {img_path}: {e}')
    except Exception as e:
        logger_thread.error(f'Classification failed: {e}')
