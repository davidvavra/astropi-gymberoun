from orbit import ISS


def get_location():
    location = []
    point = ISS.coordinates()

    location.append(point.latitude.degrees)
    location.append(point.longitude.degrees)

    return(location)

