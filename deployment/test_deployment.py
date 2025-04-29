from skimage.metrics import structural_similarity as ssim
import estimate_pose as ep
import pipeline_params as pp
import numpy as np
import cv2
import os


def main():
    """
    Captures live video from the webcam and saves an image when 's' is pressed. It then outputs aruco marker rotation and translation vectors
    as well as the ssim score.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera
    count = 0
    new_set = False
    corners1, corners2 = None, None
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    # Initialize variables to store images and pose estimates
    image1_aruco, rvecs1, tvecs1, corners1, ids1 = None, None, None, None, None
    image2_aruco, rvecs2, tvecs2, corners2, ids2 = None, None, None, None, None
    print("Press 's' to save an image, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF
        # Detect ArUco markers and display on incoming frame
        undistorded_frame = cv2.undistort(frame, pp.CAMERA_MATRIX, pp.DIST_COEFF)
        currnet_corners, current_ids = ep.detect_aruco_markers(undistorded_frame)
  
        if key == ord('s'):  # Save the frame to a variable when 's' is pressed
            success, aruco_img, rvecs, tvecs, corners, ids = ep.estimate_pose(undistorded_frame, False)
            print(rvecs, tvecs, corners, ids)
            if success:
                count += 1
                print("Image captured and marker pose estimated.")
            else:
                print("No markers detected. Please try again.")
                continue
            if count%2 == 1:
                image1_aruco, rvecs1, tvecs1, corners1, ids1 = aruco_img, rvecs, tvecs, corners, ids
            elif count%2 == 0:
                image2_aruco, rvecs2, tvecs2, corners2, ids2 = aruco_img, rvecs, tvecs, corners, ids
            
            if image1_aruco is not None and image2_aruco is not None:
                # Convert both images to grayscale
                image1_aruco_gray = cv2.cvtColor(image1_aruco, cv2.COLOR_BGR2GRAY)
                image2_aruco_gray = cv2.cvtColor(image2_aruco, cv2.COLOR_BGR2GRAY)
                ssim_score, diff_frame = ssim(image1_aruco_gray, image2_aruco_gray, full=True)
                print(f"SSIM Score: {ssim_score}")
                # Compare rotation vectors (rvecs) and translation vectors (tvecs)
                t_diff, angle_diff = ep.compare_pose(
                    np.array(rvecs1), np.array(tvecs1), np.array(rvecs2), np.array(tvecs2)
                )
                print(f"Translation Difference: {t_diff:.4f} meters")
                print(f"Rotation Difference: {angle_diff:.4f} degrees")
             
                # Create the directory path for saving files
                output_dir = f"test_deployment/test_{count-1}"
                os.makedirs(output_dir, exist_ok=True)

                # Save images
                # Normalize the difference frame for visualization
                diff_frame_normalized = cv2.normalize(diff_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(f"{output_dir}/image1.png", image1_aruco)
                cv2.imwrite(f"{output_dir}/image2.png", image2_aruco)
                cv2.imwrite(f"{output_dir}/diff_normalized.png", diff_frame_normalized)

                # Save translation and rotation differences to a text file
                with open(f"{output_dir}/differences.txt", "w") as file:
                    file.write(f"SSIM Score: {ssim_score}\n")
                    file.write(f"Translation Difference: {t_diff:.4f} meters\n")
                    file.write(f"Rotation Difference: {angle_diff:.4f} degrees\n")
        elif key == ord('q'):  # Quit the program when 'q' is pressed
                    print("Exiting...")
                    break
        
        # Display the frame with detected markers
        cv2.aruco.drawDetectedMarkers(undistorded_frame, currnet_corners, current_ids)
        if corners1 is not None and count %2 == 1:
            cv2.aruco.drawDetectedMarkers(undistorded_frame, corners1, ids1)
        if corners2 is not None and count %2 == 0:
            cv2.aruco.drawDetectedMarkers(undistorded_frame, corners2, ids2)
        cv2.imshow("Aruco Marker Detection", undistorded_frame)

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()