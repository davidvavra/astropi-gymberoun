import os
import logging
logger = logging.getLogger("astropi.create_folders")

def create_folder(base_dir):
    """Function to create directories for storing and organizing all the images

    Args:
        base_dir (str): path to directory in which all other subdirecotries shall be created

    Returns:
        tuple(bool, string, Exception): Bool False if something happened, string where and Exception why, else (True, "all", None)
    """
    # Create images folder
    try:
        if not(os.path.isdir(os.path.join(base_dir, "images"))):
            os.mkdir(os.path.join(base_dir, "images"))
            logger.info(f"Created directory 'images'")
        else:
            logger.info(f"Directory 'images' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'images' directory")
        return (False, "images", E)

    # Create images/raw folder
    try:
        if not(os.path.isdir(os.path.join(base_dir, "images/raw"))):
            os.mkdir(os.path.join(base_dir, "images/raw"))
            logger.info(f"Created directory 'images/raw'")
        else:
            logger.info(f"Directory 'images/raw' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'images/raw' directory")
        return (False, "images/raw", E)

    # Create images/cropped folder
    try:
        if not(os.path.isdir(os.path.join(base_dir, "images/cropped"))):
            os.mkdir(os.path.join(base_dir, "images/cropped"))
            logger.info(f"Created directory 'images/cropped'")
        else:
            logger.info(f"Directory 'images/cropped' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'images/cropped' directory")
        return (False, "images/cropped", E)
    
    # Create images/masked folder
    try:
        if not(os.path.isdir(os.path.join(base_dir, "images/masked"))):
            os.mkdir(os.path.join(base_dir, "images/masked"))
            logger.info(f"Created directory 'images/masked'")
        else:
            logger.info(f"Directory 'images/masked' already exists.")
    except Exception as E:
        logger.error(f"Exception {E} while creating 'images/masked' directory")
        return (False, "images/masked", E)
    
    #Everything went fine
    return (True, "all", None)