import os

def create_folder(base_folder):
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
    except Exception as E:
        return (False, "images", E)

    # Create images/raw folder
    try:
        if not(os.path.isdir(os.path.join(base_dir, "images/raw"))):
            os.mkdir(os.path.join(base_dir, "images/raw"))
    except Exception as E:
        return (False, "images/raw", E)

    # Create images/cropped folder
    try:
        if not(os.path.isdir(os.path.join(base_dir, "images/cropped"))):
            os.mkdir(os.path.join(base_dir, "images/cropped"))
    except Exception as E:
        return (False, "images/cropped", E)
    
    # Create images/masked folder
    try:
        if not(os.path.isdir(os.path.join(base_dir, "images/masked"))):
            os.mkdir(os.path.join(base_dir, "images/masked"))
    except Exception as E:
        return (False, "images/masked", E)
    
    #Everything went fine
    return (True, "all", None)