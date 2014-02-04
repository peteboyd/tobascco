import numpy as np

def calc_angle(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = v1[:3] / np.linalg.norm(v1[:3])
    v2_u = v2[:3] / np.linalg.norm(v2[:3])
    #Note: using the test (v1_u==v2_u).all(), which returns
    # true if all values of v1_u are equal to v2_u fails here.
    # there must be a rounding error.
    if np.allclose(v1_u, v2_u): 
        return 0.0
    elif np.allclose(v1_u, -v2_u): 
        return np.pi 
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if np.allclose(v1_u, v2_u):
            return 0.0
        else:
            return np.pi
    return angle

def rotation_matrix(axis, angle, point=None):
    """
    returns a 3x3 rotation matrix based on the
    provided axis and angle
    """
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.)
    b, c, d = -axis*np.sin(angle / 2.)

    R = np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
              [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
              [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])

    M = np.identity(4)
    M[:3,:3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3,3] = point - np.dot(R, point)
    return M
