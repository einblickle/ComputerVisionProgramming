import numpy as np
import matplotlib.pyplot as plt
import transforms3d as trafo3d


# define grid
def makeGridPoints(halfWidth, spacing, pointSpacing):

    pointsOnLine = np.arange(-halfWidth, halfWidth, pointSpacing)
    majorLines = np.arange(-halfWidth, halfWidth, spacing)

    all_points_x = []
    all_points_y = []
    # make x Grid
    for value in majorLines:
        # generate the coordinate values
        all_points_x.extend([value]*len(pointsOnLine))
        all_points_y.extend(pointsOnLine)

    # swap coordinates to create grid
    xP = all_points_x.copy()
    yP = all_points_y.copy()
    

    all_points_x.extend(yP)
    all_points_y.extend(xP)
    all_points_z = [0]*len(all_points_x)
    
    return all_points_x, all_points_y, all_points_z


def pol2cart(r, theta, phi):
    return  r * np.sin(theta) * np.cos(phi), r *np.sin(theta) * np.sin(phi), r * np.cos(theta)

def makeSpherePoints(center_x, center_y, center_z, radius):
    theta = np.linspace(0, np.pi, 90)
    
    phi = np.linspace(-np.pi, np.pi, 150)
    
    all_points_theta = []
    all_points_phi = []
    for val in phi:
        all_points_phi.extend([val]*len(theta))
        all_points_theta.extend(theta)

    # to cartesian coordinates
    all_points_x = []
    all_points_y = []
    all_points_z = []
    for index in range(len(all_points_theta)):
        xi, yi, zi = pol2cart(radius, all_points_theta[index], all_points_phi[index])
        all_points_x.extend([xi+center_x])
        all_points_y.extend([yi+center_y])
        all_points_z.extend([zi+center_z])
    
    return all_points_x, all_points_y, all_points_z


def getIntrinsicCameraCalibrationMatrix(focalLength, c_x, c_y):
    intCameraMatrix = np.array([
        [focalLength, 0, 0, c_x],
        [0, focalLength, 0, c_y],
        [0,0,1,0]
        ])
    return intCameraMatrix

def getExtrinsicCameraCalibrationMatrix(rotAngle_x, rotAngle_y, rotAngle_z, t_x, t_y, t_z):
    tranfoMat = translation_matrix(t_x, t_y, t_z) @ rotation_matrix_z(rotAngle_z) @ rotation_matrix_y(rotAngle_y) @ rotation_matrix_x(rotAngle_x)
    return tranfoMat

def hom2Cart(coordinates):
    cart = coordinates/coordinates[-1,:]
    return cart[:-1,:]



def translation_matrix(t_x, t_y, t_z):
    transMat = np.array([
        [1,0,0,t_x],
        [0,1,0,t_y],
        [0,0,1,t_z],
        [0,0,0,1]
        ])
    return transMat


def rotation_matrix_z(alpha_degree):
    alpha_radian = np.deg2rad(alpha_degree)

    rotMat =np.array([
    [np.cos(alpha_radian), -np.sin(alpha_radian), 0, 0],
    [np.sin(alpha_radian), np.cos(alpha_radian), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])
    return rotMat


def rotation_matrix_y(alpha_degree):
    alpha_radian = np.deg2rad(alpha_degree)

    rotMat = np.array([
    [np.cos(alpha_radian), 0, np.sin(alpha_radian), 0],
    [0, 1, 0, 0],
    [-np.sin(alpha_radian), 0, np.cos(alpha_radian), 0],
    [0, 0, 0, 1]])
    return rotMat



def rotation_matrix_x(alpha_degree):
    alpha_radian = np.deg2rad(alpha_degree)

    rotMat = np.array([
        [1, 0, 0, 0],
        [0, np.cos(alpha_radian), -np.sin(alpha_radian), 0],
        [0, np.sin(alpha_radian), np.cos(alpha_radian), 0],
        [0, 0, 0, 1]
    ])
    return rotMat





    if out is None:
        data = numpy.array(data, dtype=numpy.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(numpy.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = numpy.array(data, copy=False)
        data = out
    length = numpy.atleast_1d(numpy.sum(data*data, axis))
    numpy.sqrt(length, length)
    if axis is not None:
        length = numpy.expand_dims(length, axis)
    data /= length
    if out is None:
        return data



if __name__ == "__main__":
    plt.close('all')

    pt = np.array([0,0,0,1])
    pt = pt[:,np.newaxis]
    
    xP, yP, zP = makeGridPoints(500, 50, 1)
    pointsGrid = np.array([xP, yP, zP,[1]*len(zP)])

    xP,yP,zP = makeSpherePoints(0,0,0, 100)
    pointsSphere1 = np.array([xP, yP, zP,[1]*len(zP)])

    xP,yP,zP = makeSpherePoints(-125,0,-600,100)
    pointsSphere2 = np.array([xP, yP, zP,[1]*len(zP)])

    xP,yP,zP = makeSpherePoints(250,0,-250,100)
    pointsSphere3 = np.array([xP, yP, zP,[1]*len(zP)])


    
    intCamMat = getIntrinsicCameraCalibrationMatrix(10, 0,0)
    angle =70
    angleRad=np.deg2rad(angle)
    dist = 1000
    extCamMat = getExtrinsicCameraCalibrationMatrix(angle,0,0,0,0, dist)


    projectedPtHom = intCamMat@extCamMat@pt
    projectedPt = hom2Cart(projectedPtHom)
    
    projectedPointsGridHom = intCamMat@extCamMat@pointsGrid
    projectedPointsGrid = hom2Cart(projectedPointsGridHom)

    projectedPointsShpere1Hom = intCamMat@extCamMat@pointsSphere1
    projectedPointsSphere1 = hom2Cart(projectedPointsShpere1Hom)

    projectedPointsShpere2Hom = intCamMat@extCamMat@pointsSphere2
    projectedPointsSphere2 = hom2Cart(projectedPointsShpere2Hom)

    projectedPointsShpere3Hom = intCamMat@extCamMat@pointsSphere3
    projectedPointsSphere3 = hom2Cart(projectedPointsShpere3Hom)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(pointsGrid[0,:], pointsGrid[1,:], c = 'black', s=2)
    plt.scatter(pt[0,:], pt[1,:], c='r')
    ax.set_aspect('equal')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(pointsSphere1[2,:], pointsSphere1[0,:], s=1)
    ax.set_aspect('equal')
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.scatter(projectedPointsGrid[0,:], projectedPointsGrid[1,:], c='black', s=2)
    plt.scatter(projectedPointsSphere1[0,:], projectedPointsSphere1[1,:], c='b', s=1)
    plt.scatter(projectedPointsSphere2[0,:], projectedPointsSphere2[1,:], c='c', s=1)
    plt.scatter(projectedPointsSphere3[0,:], projectedPointsSphere3[1,:], c='g', s=1)
    plt.scatter(projectedPt[0,:], projectedPt[1,:], c='r', s=4)
    plt.show()