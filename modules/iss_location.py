from orbit import ISS


def get():
    """Function which gets current location of ISS

    Returns:
        [] - latitude, longitude
    """

    location = []
    point = ISS.coordinates()
    location.append(point.latitude.degrees)
    location.append(point.longitude.degrees)
    return location
