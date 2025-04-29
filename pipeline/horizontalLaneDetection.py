import cv2
import numpy as np

# TODO Try dilating and eroding the image to remove noise and fill gaps in the contours

morphological_kernel = (5, 5)  # Kernel size for morphological operations
iterations = 3  # Number of iterations for dilation and erosion
grad_x_scale = 1 # Scale for Sobel X gradient
grad_y_scale = 2 # Scale for Sobel Y gradient
sobel_ksize = 3 # Kernel size for Sobel operator
binary_threshold = 225 # Threshold for binary image
contour_width_ratio = .3 # Minimum width of contour to be considered a track lane contour. This is a percentage of the frame width 
canny_min = 150 # First threshold for Canny edge detection
canny_max = 200 # Second threshold for Canny edge detection

def horizontalLaneDetection(blurred):
    '''
    This function detects horizontal track lanes in a given image. It assumes that no motion is detected in the blurred frame passed in,
    implying that there are currently no athletes on the track.  

    Args: 
        blurred (numpy.ndarray): Blurred frame with no motion detected (no athletes on the track)
    Returns:
        horizontal_line_contours (list): List of contours representing horizontal track lanes
    '''
    # undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    # roi = (0, 300, 1920, 400)  # Define the region of interest (x, y, width, height)
    # cropped_frame = undistorted_frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    # Convert to grayscale
    # gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    # blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    # dilate and erode the image to remove noise and fill gaps in the contours

    blurred = cv2.dilate(blurred, None, iterations)
    blurred = cv2.erode(blurred, None, iterations)
    edge = cv2.Canny(blurred, canny_min, canny_max)
    # Sobel gradients
    grad_x = cv2.Sobel(edge, cv2.CV_16S, 1, 0, ksize=sobel_ksize, scale=grad_x_scale)
    grad_y = cv2.Sobel(edge, cv2.CV_16S, 0, 1, ksize=sobel_ksize, scale=grad_y_scale)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    # Subtract gradients to ensure horizontal lines are highlighted and vertical lines are suppressed
    diff_frame = cv2.subtract(abs_grad_y, abs_grad_x)
    _, max_val, _, _ = cv2.minMaxLoc(diff_frame)
    if max_val == 0:
        return []
    # Scale and threshold
    scaled_sobely = cv2.convertScaleAbs(diff_frame, alpha=255.0 / max_val)
    _, binary_edges = cv2.threshold(scaled_sobely, binary_threshold, 255, cv2.THRESH_BINARY)
    # Morphological closing
    kernel = np.ones(morphological_kernel, np.uint8)
    closed_frame = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)

    # Release the video writer
    contours, _ = cv2.findContours(closed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_line_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > contour_width_ratio * 960: # Check if the contour width is greater than the threshold
            horizontal_line_contours.append(contour)

    return horizontal_line_contours

def fit_poly(frame, viz_frame, lane_contours, visualize):
    y_values = []
    for i, contour in enumerate(lane_contours):
            # Extract x and y coordinates from the contour
            contour_points = contour[:, 0, :]
            x_contour = contour_points[:, 0]
            y_contour = contour_points[:, 1]

            # Fit a 2nd-degree polynomial
            if len(x_contour) > 2:  # Ensure there are enough points to fit a polynomial
                poly_coeffs = np.polyfit(x_contour, y_contour, 2)
                poly_func = np.poly1d(poly_coeffs)

            # Generate points for the polynomial curve across the entire frame width
            x_poly = np.linspace(0, frame.shape[1] - 1, 960)
            y_poly = poly_func(x_poly)
            y_row = []  # Initialize y_row before appending values
            for y in y_poly:
                if 0 <= y < frame.shape[0]:
                    y_row.append(int(y))
                else:
                    y_row.append(None)
            y_values.append(y_row)
            # Draw the polynomial curve on the frame
            if visualize:
                for j in range(len(x_poly) - 1):
                    pt1 = (int(x_poly[j]), int(y_poly[j]))
                    pt2 = (int(x_poly[j + 1]), int(y_poly[j + 1]))
                    if 0 <= pt1[1] < frame.shape[0] and 0 <= pt2[1] < frame.shape[0]:
                        cv2.line(viz_frame, pt1, pt2, (0, 255, 255), 2)

    return y_values