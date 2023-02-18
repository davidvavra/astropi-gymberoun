from orbit import ISS


def get():
    location = []
    point = ISS.coordinates()
    location.append(point.latitude.degrees)
    location.append(point.longitude.degrees)
    return location
