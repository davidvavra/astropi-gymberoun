import logging
import os

from modules import files


def create_logger(base_folder):
    """
    Creates a logger which writes everything to a file and error+ logs to error output.

    Args:
        base_folder: base folder for everything
    """
    logger = logging.getLogger("astropi")
    try:
        logger.setLevel(logging.DEBUG)
        folder_path = f"{base_folder}/{files.LOGS_FOLDER}"
        if not (os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        fh = logging.FileHandler(f"{folder_path}/main.log")
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
        print(f"Couldn't create logfile: ${E}")
    return logger
