import logging


def create_logger(base_folder):
    logger = logging.getLogger("astropi")
    try:
        logger.setLevel(logging.DEBUG)
        # create file handler which logs.py even debug messages
        fh = logging.FileHandler(base_folder/"logs/main.log")
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
    except:
        print("Couldn't create logfile")
    return logger
