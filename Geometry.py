import numpy as np


def get_Direction(x, y):
    if len(y.shape) == 1:
        return (y-x)/np.linalg.norm(y-x)
    else:
        return (y-x)/np.linalg.norm(y-x, axis=1)[:, np.newaxis]


def get_Perpendicular(v):
    if v[0] == v[1] == v[2] == 0:
        raise ValueError('Error: Direction vector is 0.')

    if v[0] == 0:
        return np.array([1, 0, 0])
    if v[1] == 0:
        return np.array([0, 1, 0])
    if v[2] == 0:
        return np.array([0, 0, 1])

    a = np.array([0.5, 0.5, -0.5*(v[0] + v[1])/v[2]])

    return a/np.linalg.norm(a)


def get_VectorAngle(v):
    x, y, z = v

    fi = np.arctan(y/x)
    th = np.arctan(z/np.sqrt(x**2+y**2))

    if (x > 0) and (y > 0):
        return fi, th
    elif (x < 0) and (y > 0):
        return fi+np.pi, th
    elif (x < 0) and (y < 0):
        return fi+np.pi, th
    elif (x > 0) and (y < 0):
        return fi+2*np.pi, th


def get_DeltaAngle(x1, x2, x3):
    d1 = get_Direction(x2, x1)
    d2 = get_Direction(x3, x2)

    fi1, th1 = get_VectorAngle(d1)
    fi2, th2 = get_VectorAngle(d2)

    dfi, dth = np.abs(fi1-fi2), np.abs(th1-th2)

    if dfi > np.pi:
        return 2*np.pi-dfi, dth
    else:
        return dfi, dth


def get_Intersection(x1, x2, xC, R):
    dir = get_Direction(x1, x2)
    a = np.linalg.norm(dir)**2
    b = 2*np.dot(dir, x2 - xC)
    c = get_Distance(x2, xC)**2 - R**2

    D = b**2 - 4*a*c

    if D <= 0:
        return False
    else:
        return True


def get_Distance(coord, c):
    d, D = coord.shape, c.shape
    if len(d) == 1:
        if D[0] == 2 and d[0] == 2:
            return np.sqrt((coord[0]-c[0])**2 + (coord[1]-c[1])**2)
        elif D[0] == 3 and d[0] == 3:
            return np.sqrt((coord[0]-c[0])**2 + (coord[1]-c[1])**2 + (coord[2]-c[2])**2)
        else:
            print ("Wrong dimensions")

    elif len(d) == 2:
        if D[0] == 2 and d[1] == 2:
            return np.sqrt((coord[:, 0]-c[0])**2 + (coord[:, 1]-c[1])**2)
        elif D[0] == 3 and d[1] == 3:
            return np.sqrt((coord[:, 0]-c[0])**2 + (coord[:, 1]-c[1])**2 + (coord[:, 2]-c[2])**2)
        else:
            print ("Wrong dimensions")
    else:
        print ("Wrong dimensions")
