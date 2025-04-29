from skimage.metrics import structural_similarity as ssim
import estimate_pose as ep
import pipeline_params as pp
import numpy as np
import cv2
import os
import time

WAIT_TIME = 2  # seconds
ideal_rvecs, ideal_tvecs, ideal_corners, ideal_ids = None, None, None, None

def update_key(key):
    current_drone = key
    global ideal_rvecs, ideal_tvecs, ideal_corners, ideal_ids
    ideal_rvecs = pp.ARUCO_MARKERS[current_drone]["ideal_rvecs"]
    ideal_tvecs = pp.ARUCO_MARKERS[current_drone]["ideal_tvecs"]
    ideal_corners = pp.ARUCO_MARKERS[current_drone]["ideal_corners"]
    ideal_ids = pp.ARUCO_MARKERS[current_drone]["ideal_ids"]

def main():
    """
    This script , when in the ideal position, saves the imformation of the aruco marker in the marker_information folder.
    """
    current_drone = None
    # Initialize variables to store images and pose estimates
    aruco_img, rvecs, tvecs, corners, ids = None, None, None, None, None
    last_check_time = time.time()  # Initialize the last check time
    key = 1
    update_key(key)
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        # Load in frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break        
        # Load in the ideal position of the marker
        undistorted_frame = cv2.undistort(frame, pp.CAMERA_MATRIX, pp.DIST_COEFF)
        visualize_frame = undistorted_frame.copy()
        # Draw ideal position of the marker in red
        cv2.aruco.drawDetectedMarkers(visualize_frame, ideal_corners, ideal_ids, borderColor=(0, 0, 255))
        cv2.drawFrameAxes(visualize_frame, pp.CAMERA_MATRIX, pp.DIST_COEFF, ideal_rvecs, ideal_tvecs, pp.MARKER_SIZE / 2)
        # # the detect markers and do pnp pose estimation
        success, aruco_img, rvecs, tvecs, corners, ids = ep.estimate_pose(undistorted_frame, visualize=False)
        cv2.aruco.drawDetectedMarkers(visualize_frame, corners, ids, borderColor=(255, 0, 0))
        cv2.drawFrameAxes(visualize_frame, pp.CAMERA_MATRIX, pp.DIST_COEFF, ideal_rvecs, ideal_tvecs, pp.MARKER_SIZE / 2)
        if success:
            # Compare the current position with the ideal position
            t_diff, angle_diff = ep.compare_pose(
                np.array(rvecs), np.array(tvecs), np.array(ideal_rvecs), np.array(ideal_tvecs))
            current_time = time.time()
            if current_time - last_check_time >= WAIT_TIME:  # Check if 5 seconds have passed
                last_check_time = current_time  # Update the last check time
                print(f"Translation Difference: {t_diff:.4f} meters")
                print(f"Rotation Difference: {angle_diff:.4f} degrees")
        cv2.imshow("Undistorted frame", visualize_frame)
        key = cv2.waitKey(1) & 0xFF  # Capture key press
        if key == ord('q'):  # Press 'q' to exit the loop
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            update_key(int(chr(key)))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()