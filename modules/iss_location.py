from orbit import ISS
import logging
logger = logging.getLogger("astropi")


def get():
    """Function which gets current location of ISS

    Returns:
        [] - latitude, longitude
    """
    location = []
    point = ISS.coordinates()
    location.append(point.latitude.degrees)
    location.append(point.longitude.degrees)
    logger.info(f'Calculated ISS location: latitude {location[0]}, longitude {location[1]}')
    return location
