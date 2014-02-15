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
    
def central_moment(weights, vects, mean):
    """Obtain the central moments"""
    mx, my, mz = mean
    dic={}
    def moment(l,m,n):
        try:
            return dic[(l,m,n)]
        except KeyError:
            mom = 0.
            for ind, (x,y,z) in enumerate(vects):
                mom += ((x-mx)**l)*((y-my)**m)*((z-mz)**n)*weights[ind]
            dic[(l,m,n)] = mom
            return mom
    return moment

def raw_moment(weights, vects):
    dic = {}
    def moment(l, m, n):
        try:
            return dic[(l,m,n)]
        except KeyError:
            mom = 0.
            for ind, (x,y,z) in enumerate(vects):
                mom += (x**l)*(y**m)*(z**n)*weights[ind]
            dic[(l,m,n)] = mom
            return mom
    return moment

def elipsoid_vol(cm):
    mat = np.matrix([[cm(2,0,0), cm(1,1,0), cm(1,0,1)],
                     [cm(1,1,0), cm(0,2,0), cm(0,1,1)],
                     [cm(1,0,1), cm(0,1,1), cm(0,0,2)]])
    vol = (np.pi*4./3.*np.linalg.det(mat))**(1./3.)
    return vol

def r_gyr(cm):
    return np.sqrt((cm(2,0,0)+cm(0,2,0)+cm(0,0,2))/(3.*cm(0,0,0)))

def get_CI(cm):
    r = r_gyr(cm)
    s3 = 1./((cm(0,0,0)**3)*r**9)
    s4 = 1./((cm(0,0,0)**4)*r**9)
    # second order
    a1 = cm(0,0,2) - cm(0,2,0)
    a2 = cm(0,2,0) - cm(2,0,0)
    a3 = cm(2,0,0) - cm(0,0,2)
    # third order
    b1 = cm(0,2,1) - cm(2,0,1)
    b2 = cm(1,0,2) - cm(1,2,0)
    b3 = cm(2,1,0) - cm(0,1,2)
    b4 = cm(0,0,3) - cm(2,0,1) - 2.*cm(0,2,1)
    b5 = cm(0,0,3) - cm(2,0,1) - 2.*cm(0,2,1)
    b6 = cm(3,0,0) - cm(1,2,0) - 2.*cm(1,0,2)
    b7 = cm(0,2,1) - cm(0,0,3) + 2.*cm(2,0,1)
    b8 = cm(1,0,2) - cm(3,0,0) + 2.*cm(1,2,0)
    b9 = cm(2,1,0) - cm(0,3,0) + 2.*cm(0,1,2)
    b10 = cm(0,2,1) + cm(2,0,1) - 3.*cm(0,0,3)
    b11 = cm(0,1,2) + cm(2,1,0) - 3.*cm(0,3,0)
    b12 = cm(1,0,2) + cm(1,2,0) - 3.*cm(3,0,0)
    b13 = cm(0,2,1) + cm(0,0,3) + 3.*cm(2,0,1)
    b14 = cm(1,0,2) + cm(3,0,0) + 3.*cm(1,2,0)
    b15 = cm(2,1,0) + cm(0,3,0) + 3.*cm(0,1,2)
    b16 = cm(0,1,2) + cm(0,3,0) + 3.*cm(2,1,0)
    b17 = cm(2,0,1) + cm(0,0,3) + 3.*cm(0,2,1)
    b18 = cm(1,2,0) + cm(3,0,0) + 3.*cm(1,0,2)
    #fourth order
    g1 = cm(0,2,2) - cm(4,0,0)
    g2 = cm(2,0,2) - cm(0,4,0)
    g3 = cm(2,2,0) - cm(0,0,4)
    g4 = cm(1,1,2) + cm(1,3,0) + cm(3,1,0)
    g5 = cm(1,2,1) + cm(1,0,3) + cm(3,0,1)
    g6 = cm(2,1,1) + cm(0,1,3) + cm(0,3,1)
    g7 = cm(0,2,2) - cm(2,2,0) + cm(0,0,4) - cm(4,0,0)
    g8 = cm(2,0,2) - cm(0,2,2) + cm(4,0,0) - cm(0,4,0)
    g9 = cm(2,2,0) - cm(2,0,2) + cm(0,4,0) - cm(0,0,4)

    CI = 4.*s3*(cm(1,1,0)*(cm(0,2,1)*(3.*g2-2.*g3-g1) -
                           cm(2,0,1)*(3.*g1-2.*g3-g2) + b12*g5 -
                           b11*g6 + cm(0,0,3)*g8) + 
                cm(1,0,1)*(cm(2,1,0)*(3.*g1-2.*g2-g3) -
                           cm(0,1,2)*(3.*g3-2.*g2-g1)+b10*g6-b12*g4 +
                           cm(0,3,0)*g7) + 
                cm(0,1,1)*(cm(1,0,2)*(3.*g3-2.*g1-g2)-
                           cm(1,2,0)*(3.*g2-2.*g1-g3) + 
                           b11*g4-b10*g5+cm(3,0,0)*g9) + 
                cm(0,0,2)*(b18*g6-b15*g5-2.*(cm(1,1,1)*g8+b1*g4))+
                cm(0,2,0)*(b17*g4-b14*g6-2.*(cm(1,1,1)*g7+b3*g5))+
                cm(2,0,0)*(b16*g5-b13*g4-2.*(cm(1,1,1)*g9+b2*g6))) - \
        16.*s4*(cm(0,1,1)*a2*a3*b2+cm(1,0,1)*a1*a2*b3 +
                cm(1,1,0)*a1*a3*b1-cm(1,1,1)*a1*a2*a3 -
                cm(0,1,1)*cm(0,1,1)*(cm(1,1,1)*a1-cm(0,1,1)*b2-cm(1,0,1)*b5-cm(1,1,0)*b7) -
                cm(1,0,1)*cm(1,0,1)*(cm(1,1,1)*a3-cm(1,0,1)*b3-cm(1,1,0)*b4-cm(0,1,1)*b8) -
                cm(1,1,0)*cm(1,1,0)*(cm(1,1,1)*a2-cm(1,1,0)*b1-cm(0,1,1)*b6-cm(1,0,1)*b9) +
                cm(0,1,1)*cm(0,1,1)*(cm(0,0,2)*b1+cm(0,2,0)*b4+cm(2,0,0)*b7) +
                cm(0,1,1)*cm(1,1,0)*(cm(0,2,0)*b3+cm(2,0,0)*b5+cm(0,0,2)*b9) +
                cm(1,0,1)*cm(1,0,1)*(cm(2,0,0)*b2+cm(0,0,2)*b6+cm(0,2,0)*b8))

    return CI

