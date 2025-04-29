import cv2
import numpy as np
import pipeline_params as pp
# Currently assumes only one marker is in frame

def detect_aruco_markers(image):
    '''
    This function detects ArUco markers in the input image and returns their corners and IDs.

    Args:
        image (numpy.ndarray): Input image containing ArUco markers.
    
    Returns:
        corners (list): List of detected marker corners.
        ids (numpy.ndarray): Array of detected marker IDs.
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create an ArUco detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Detect the markers in the image
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    return corners, ids

# Implement pnp funcitoniwth aruco markers
def estimate_pose(image, visualize = False):
    '''
    This function estimates the pose of ArUco markers in the input image. Assumes only one marker in frame
    Args:
        image (numpy.ndarray): Input image containing ArUco markers.
    Returns:
        image (numpy.ndarray): Output image with detected markers and axes drawn.
        rvecs (list): List of rotation vectors for each detected marker.
        tvecs (list): List of translation vectors for each detected marker.
    '''
    corners, ids = detect_aruco_markers(image)
    # Define the real world coordinates for the corners of the markers
    marker_length = 1  # Here, 1m is the length of the marker's side
    obj_points = np.array([
        [-marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ])

    rvecs, tvecs = [], []

    if ids is not None:
        for i in range(len(ids)):
            # Estimate pose of each marker
            ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], pp.CAMERA_MATRIX, pp.DIST_COEFF)
            if ret:
                rvecs.append(rvec)
                tvecs.append(tvec)
                # Draw axis for each marker
                if visualize:
                    cv2.drawFrameAxes(image, pp.CAMERA_MATRIX, pp.DIST_COEFF, rvec, tvec, marker_length / 2)
                    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    if ids is None or len(ids) == 0:
        success = False
    else:
        success = True

    return success, image, rvecs, tvecs, corners, ids

def compare_pose(rvec1, tvec1, rvec2, tvec2):
    '''
    This function compares the pose of two ArUco markers by calculating the translation and rotation differences.

    Args:
        rvec1 (numpy.ndarray): Rotation vector of the first marker.
        tvec1 (numpy.ndarray): Translation vector of the first marker.
        rvec2 (numpy.ndarray): Rotation vector of the second marker.
        tvec2 (numpy.ndarray): Translation vector of the second marker.
    
    Returns:
        t_diff (float): Translation difference between the two markers.
        angle_diff (float): Rotation difference between the two markers in degrees.
    '''

    if rvec1 is None or rvec2 is None:
        print("Marker not detected in one or both images.")
        return None, None

    # Compute translation difference
    t_diff = np.linalg.norm(tvec1 - tvec2)  # Euclidean distance

    # Convert rotation vectors to rotation matrices
    R1, _ = cv2.Rodrigues(rvec1.reshape(3, 1))
    R2, _ = cv2.Rodrigues(rvec2.reshape(3, 1))

    # Compute relative rotation matrix
    R_diff = R1 @ R2.T  # R1 * R2^(-1)

    # Compute angle difference in degrees
    # Compute angle difference in degrees
    trace_value = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)  # Ensure value is within valid range for arccos
    angle_diff = np.arccos(trace_value) * (180 / np.pi)

    # Compute roll, pitch, and yaw differences
    # if np.isclose(np.linalg.det(R_diff), 1.0):  # Check if R_diff is a valid rotation matrix
    #     roll_diff = np.arctan2(R_diff[2, 1], R_diff[2, 2]) * (180 / np.pi)
    #     pitch_diff = np.arctan2(-R_diff[2, 0], np.sqrt(R_diff[2, 1]**2 + R_diff[2, 2]**2)) * (180 / np.pi)
    #     yaw_diff = np.arctan2(R_diff[1, 0], R_diff[0, 0]) * (180 / np.pi)
    # else:
    #     roll_diff, pitch_diff, yaw_diff = None, None, None  # Handle invalid rotation matrix case

    # return t_diff, angle_diff, roll_diff, pitch_diff, yaw_diff

    return t_diff, angle_diff


