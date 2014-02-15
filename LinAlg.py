import numpy as np

RAD2DEG = 180./np.pi 
DEG2RAD = np.pi/180.

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

def calc_axis(v1, v2):
    v1_u = v1[:3] / np.linalg.norm(v1[:3])
    v2_u = v2[:3] / np.linalg.norm(v2[:3])
    if np.allclose(v1_u, v2_u) or \
                np.allclose(v1_u, -v2_u): 
        return np.array([1., 0., 0.])
    a = np.cross(v1_u, v2_u)
    return a / np.linalg.norm(a)

def rotation_from_omega(w):

    theta = np.linalg.norm(w)
    omega_x = np.array([[   0., -w[2],  w[1]],
                        [ w[2],    0., -w[0]],
                        [-w[1],  w[0],    0.]])
    R = np.identity(4)
    M = np.identity(3) + np.sin(theta)/theta*omega_x + \
            (1. - np.cos(theta))/(theta*theta) * np.linalg.matrix_power(omega_x, 2)
    R[:3,:3] = M
    return R

def rotation_from_vectors(v1, v2, point=None):
    """Obtain rotation matrix from sets of vectors.
    the original set is v1 and the vectors to rotate
    to are v2.

    """

    # v2 = transformed, v1 = neutral
    ua = np.array([np.mean(v1.T[0]), np.mean(v1.T[1]), np.mean(v1.T[2])])
    ub = np.array([np.mean(v2.T[0]), np.mean(v2.T[1]), np.mean(v2.T[2])])

    Covar = np.dot((v2 - ub).T, (v1 - ua))

    u, s, v = np.linalg.svd(Covar)
    uv = np.dot(u,v)
    d = np.identity(3) 
    d[2,2] = np.linalg.det(uv) # ensures non-reflected solution
    M = np.dot(np.dot(u,d), v)
    R = np.identity(4)
    R[:3,:3] = M
    if point is not None:
        R[:3,:3] = point - np.dot(M, point)
    return R

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
