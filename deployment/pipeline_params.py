import numpy as np

# Constants
MARKER_SIZE = 0.508  # 0.508 meters (20 inches)

# Camera Calibration Parameters
MARKER_SIZE = 0.508 # 0.508 meters (20 inches)
# Laptop Camera Calibration Parameters
# CAMERA_MATRIX  = np.array([[937.34221354,   0,  637.02054476], 
#                                [  0, 938.19746445, 341.14487524], 
#                                [0, 0, 1]], dtype=np.float32)
# DIST_COEFF = np.array([ 0.00462975, -0.25859172, -0.00902691,  0.00252958,  0.81845151],dtype=np.float32)

# Drone 1 camera calibration parameters
CAMERA_MATRIX = np.array([
    [2.80838381e+03, 0, 1.91024134e+03],
    [0, 2.81144960e+03, 1.07297946e+03],
    [0, 0, 1]
], dtype=np.float64)

DIST_COEFF = np.array([-0.05574808, 0.22661742, -0.00186, 0.00075089, -0.28986981], dtype=np.float64)

# ArUco Marker Configurations
ARUCO_MARKERS = {
    1: {
        "ideal_rvecs": np.array([[ 2.18689092],
       [-2.114056  ],
       [-0.19028711]], dtype=np.float32),
        "ideal_tvecs": np.array([[-21.84121341],
       [ -5.28203885],
       [ 65.19233797]], dtype=np.float32),
        "ideal_corners": [
            np.array([[[953., 866.],
        [954., 824.],
        [994., 824.],
        [993., 868.]]], dtype=np.float32)],
        "ideal_ids": np.array([0], dtype=np.int32),
    },
    2: {
        "ideal_rvecs": np.array([[ 3.21339722],
                                [-0.25206566],
                                [-0.00517586]], dtype=np.float32),
        "ideal_tvecs": np.array([[-10.17657145],
                                [ -5.74336829],
                                [ 20.0661713 ]], dtype=np.float32),
        "ideal_corners": [
            np.array([[[411., 209.],
                        [546., 189.],
                        [575., 328.],
                        [437., 352.]]], dtype=np.float32)],
        "ideal_ids": np.array([1], dtype=np.int32),
    },
    3: {
        "ideal_rvecs": np.array([[ 3.21339722],
                                [-0.25206566],
                                [-0.00517586]], dtype=np.float32),
        "ideal_tvecs": np.array([[-10.17657145],
                                [ -5.74336829],
                                [ 20.0661713 ]], dtype=np.float32),
        "ideal_corners": [
            np.array([[[411., 209.],
                        [546., 189.],
                        [575., 328.],
                        [437., 352.]]], dtype=np.float32)],
        "ideal_ids": np.array([2], dtype=np.int32),
    },
    4: {
        "ideal_rvecs": np.array([[ 3.21339722],
                                [-0.25206566],
                                [-0.00517586]], dtype=np.float32),
        "ideal_tvecs": np.array([[-10.17657145],
                                [ -5.74336829],
                                [ 20.0661713 ]], dtype=np.float32),
        "ideal_corners": [
            np.array([[[411., 209.],
                        [546., 189.],
                        [575., 328.],
                        [437., 352.]]], dtype=np.float32)],
        "ideal_ids": np.array([3], dtype=np.int32),
    },
    5: {
        "ideal_rvecs": np.array([[ 3.21339722],
                                [-0.25206566],
                                [-0.00517586]], dtype=np.float32),
        "ideal_tvecs": np.array([[-10.17657145],
                                [ -5.74336829],
                                [ 20.0661713 ]], dtype=np.float32),
        "ideal_corners": [
            np.array([[[411., 209.],
                        [546., 189.],
                        [575., 328.],
                        [437., 352.]]], dtype=np.float32)],
        "ideal_ids": np.array([4], dtype=np.int32),
    },
}
