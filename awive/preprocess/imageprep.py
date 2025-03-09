"""Image preparation."""
import cv2
import numpy as np


def lens_corr(img, k1=-10.0e-6, c=2, f=8.0):
    """Lens distortion correction based on lens charateristics.

    :param img: original image
    :param k1: barrel lens distortion parameter
    :param c: optical center
    :param f: focal length

    :return corr_img: image corrected for lens distortion
    """
    # define imagery charateristics
    height = img.shape[0]
    width = img.shape[1]

    # define distortion coefficient vector
    dist = np.zeros((4, 1), np.float64)
    dist[0, 0] = k1

    # define camera matrix
    mtx = np.eye(3, dtype=np.float32)

    mtx[0, 2] = width/c     # define center x
    mtx[1, 2] = height/c    # define center y
    mtx[0, 0] = f           # define focal length x
    mtx[1, 1] = f           # define focal length y

    # correct image for lens distortion
    corr_img = cv2.undistort(img, mtx, dist)

    return corr_img


def xy_coord(df):
    """
    Turn longitudes and latitudes into XY coordinates using an Equirectangular
    projection. Only applicable on a small scale.

    Input DataFrame as follows:
    df = pd.DataFrame({'lon': [a, b, c, d],
                                       'lat': [e, f, g, h]})

    Input:
    ------
    df - DataFrame containing columns with longitudes and latitudes
    PPM - Pixels per metre

    Output:
    -------
    df_new - DataFrame with xy-coordinates in metres
    """
    # set base parameters
    # r = 6378137  # meters according to WGS84
    # phi_0 = df.latitude[0]
    # cos_phi_0 = math.cos(math.radians(phi_0))

    # # create new DataFrame containing original coordinates in metres
    # df_new = pd.DataFrame()
    # df_new['x'] = [r * math.radians(lon) * cos_phi_0 for lon in df.lon.values]
    # df_new['y'] = [r * math.radians(lat) for lat in df.lat.values]

    # return df_new
    return 12


def orthorect_param(img, df_from, df_to, PPM=100, lonlat=False):
    """Image orthorectification parameters based on 4 GCPs.
    GCPs need to be at water level.

    Input DataFrames as follows:
    df = pd.DataFrame({'x': [a, b, c, d],
                                       'y': [e, f, g, h]})

    Input:
    ------
    img - original image
    df_from - DataFrame containing the xy-coordinates of the GCPs in
            the imagery in pixels
    df_to - DataFrame DataFrame containing the real xy-coordinates of
            the GCPs in metres (if in lon-lat coordinates, set lonlat=True
            to convert)
    PPM - pixels per meter in the corrected imagery (default: 100)
    lonlat - convert longitudes/latitudes to metres (default: False)

    Output:
    -------
    M_new - Transformation matrix based on image corners
    C_new - Coordinates of image corners in the orthorectified imagery
    df_to - Original coordinates of the GCPs in metres (relevant if
            converted from geographic coordinate system)
    """
    if lonlat:
        df_to = xy_coord(df_to)

    # set points to float32
    pts1 = np.float32(df_from)
    # Multiple elements inside df_to by PPM
    df_too = []
    for x in df_to:
        df_too.append(list(map(lambda x: x*PPM, x)))
    pts2 = np.float32(df_too)

    # define transformation matrix based on GCPs
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # find locations of transformed image corners
    # height, width, __ = img.shape
    height = img.shape[0]
    width = img.shape[1]
    C = np.array([[0, 0, 1],
                  [width, 0, 1],
                  [0, height, 1],
                  [width, height, 1]])
    C_new = np.array([(np.dot(i, M.T) / np.dot(i, M.T)[2])[:2] for i in C])

    C_new[:, 0] -= min(C_new[:, 0])
    C_new[:, 1] -= min(C_new[:, 1])

    # define new transformation matrix based on image corners
    # otherwise, part of the imagery will not be saved
    M_new = cv2.getPerspectiveTransform(
        np.float32(C[:, :2]),
        np.float32(C_new)
    )

    return M_new, C_new, df_to


def orthorect_trans(img, M, C):
    """
    Image orthorectification based on parameters found with
    ip.orthorect_param().

    Input:
    ------
    img - original image
    M - Transformation matrix based on image corners
    C - Coordinates of image corners in the orthorectified imagery

    Output:
    -------
    corr_img - orthorectified image
    """

    # define corrected image dimensions based on C
    cols = int(np.ceil(max(C[:, 0])))
    rows = int(np.ceil(max(C[:, 1])))

    # orthorectify image
    corr_img = cv2.warpPerspective(img, M, (cols, rows))

    return corr_img


def color_corr(img, alpha=None, beta=None, gamma=0.5):
    """
    Grey scaling, contrast- and gamma correction. Both alpha and beta need to
    be defined in order to apply contrast correction.

    Input:
    ------
    img - original image
    alpha - gain parameter for contrast correction (default: None)
    beta - bias parameter for contrast correction (default: None)
    gamma - brightness parameter for gamma correction (default: 0.5)

    Output:
    -------
    corr_img - gray scaled, contrast- and gamma corrected image
    """
    # turn image into grey scale
    corr_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if alpha and beta:
        # apply contrast correction
        corr_img = cv2.convertScaleAbs(corr_img, alpha=alpha, beta=beta)

    # apply gamma correction
    invGamma = 1./gamma
    table = (np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype('uint8'))

    corr_img = cv2.LUT(corr_img, table)

    return corr_img
