import logging
import os

from modules import files


def create_logger(base_folder):
    logger = logging.getLogger("astropi")
    try:
        logger.setLevel(logging.DEBUG)
        if not (os.path.isdir(base_folder / files.LOGS_FOLDER)):
            os.mkdir(base_folder / files.LOGS_FOLDER)
        fh = logging.FileHandler(base_folder / files.LOGS_FOLDER / "main.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
    except Exception as E:
        print("Couldn't create logfile: ${E}")
    return logger
