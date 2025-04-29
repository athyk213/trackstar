import cv2
import numpy as np
import os
import glob
import horizontalLaneDetection as hld

# Directory containing videos
video_dir = "videos/2-27/downscaled"
video_files = glob.glob(os.path.join(video_dir, "*.MP4"))

# Bounding box dimensions
bbox_width, bbox_height = 75, 100
max_contours = 6
frameSkip = 2
topCrop, bottomCrop = 150, 0

# Motion detection parameters
gaussian_kernel = (3, 3)
delta_thresh = 20
min_area = 20
max_area = 500
weight = 0.2
# Accumulated weighted average frame

# Lane detection variables
previous_avg_width= 0
num_lane_contours = 7
visualize_lane_detection = False
lane_contours = []
lane_contours_y_values = []
LANE_WIDTH_CONSTANT = 1.25
avg_lane_width = 0

wait_frames = 30

for video_path in video_files:
    print(f"Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    output_dir = os.path.join(os.path.basename(video_dir) + "-results", os.path.basename(video_path))
    os.makedirs(output_dir, exist_ok=True)
    roi_count = 0
    frame_count = 0

    contours = []
    
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % frameSkip != 0:
            continue
        frame = cv2.resize(frame, (960, 540))
        viz_frame = frame.copy()
        # Convert to grayscale and apply Gaussian blur for motion detection
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(frame, gaussian_kernel, 0)  
        # perform horizontal lane detection if no motion is detected
        lane_contours_new = hld.horizontalLaneDetection(blurred, 960)
        if len(lane_contours_new) == 0:
            print("No contours found")
            continue
        avg_width = min([cv2.boundingRect(c)[2] for c in lane_contours_new])
        if len(lane_contours_new) == num_lane_contours and avg_width>.9*previous_avg_width: # successfully detected the correct number of lane contours
            lane_contours_y_values = []  # Reset y-values for new lane contours
            previous_avg_width = avg_width
            # Calculate average y value for each contour and sort them
            lane_contours = sorted(lane_contours_new, key=lambda c: np.mean(c[:, :, 1]))
            lane_contours_y_values = [np.mean(c[:, :, 1]) for c in lane_contours]
            # Calculate average lane width based on mean y values of the lane contours
            lane_widths = []
            for i in range(len(lane_contours) - 1):
                y1 = np.mean(lane_contours[i][:, :, 1])
                y2 = np.mean(lane_contours[i + 1][:, :, 1])
                lane_widths.append(abs(y2 - y1))
            avg_lane_width = np.mean(lane_widths) if lane_widths else 0

                
        if visualize_lane_detection:
            for c in lane_contours:
                cv2.drawContours(frame, [c], -1, (255, 0, 0), 1)

        # Fit a 2D polynomial to each contour and display it on the frame

        masked_frames = []
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
            x_poly = np.linspace(0, frame.shape[1] - 1, 1000)
            y_poly = poly_func(x_poly)
            y_row = []  # Initialize y_row before appending values
            for y in y_poly:
                if 0 <= y < blurred.shape[0]:
                    y_row.append(int(y))
                else:
                    y_row.append(None)
            y_values.append(y_row)
            # Draw the polynomial curve on the frame
            for j in range(len(x_poly) - 1):
                pt1 = (int(x_poly[j]), int(y_poly[j]))
                pt2 = (int(x_poly[j + 1]), int(y_poly[j + 1]))
                if 0 <= pt1[1] < viz_frame.shape[0] and 0 <= pt2[1] < viz_frame.shape[0]:  # Ensure points are within frame bounds
                    cv2.line(viz_frame, pt1, pt2, (0, 255, 255), 2)
        
        # Gather average frame for the space between polynomial curves
        # Create masks for each lane
        for i in range(len(lane_contours) - 1):
            # Ensure the mask is single-channel (grayscale)
            mask = np.zeros(blurred.shape[:2], dtype=np.uint8)  

            for col, y_row in enumerate(zip(*y_values)):
                y1 = y_row[i]
                y2 = y_row[i + 1]
                if y1 is not None and y2 is not None:
                    cv2.line(mask, (col, y1), (col, y2), 255, 1)  # Draw white lines on the mask

            # Apply the mask correctly to an RGB image
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            masked_frame = cv2.bitwise_and(blurred, blurred, mask=mask)
            masked_frames.append(masked_frame)
            # show masked frame
            
        
              
        diff_frames = []
        avg_frames = [np.zeros_like(masked_frames[0], dtype=np.float32) for _ in range(len(masked_frames))]  # Initialize avg_frames with correct data type
        all_contours = []
        for idx, lane_frame in enumerate(masked_frames):
            # Accumulate weighted average for each lane
            if frame_count > wait_frames and idx - wait_frames >= 0 and idx - wait_frames < len(masked_frames):
                avg_frames[idx] = masked_frames[idx - wait_frames].astype(np.float32)  # Use frame from 20 frames ago
            diff_r = cv2.absdiff(cv2.convertScaleAbs(avg_frames[idx][:, :, 2]), lane_frame[:, :, 2])  # Red channel
            diff_g = cv2.absdiff(cv2.convertScaleAbs(avg_frames[idx][:, :, 1]), lane_frame[:, :, 1])  # Green channel
            diff_b = cv2.absdiff(cv2.convertScaleAbs(avg_frames[idx][:, :, 0]), lane_frame[:, :, 0])  # Blue channel

            # Combine the differences from all channels
            max_diff = max(np.max(diff_r), np.max(diff_g), np.max(diff_b))
            if max_diff != 0:
                weight_r = np.max(diff_r) / max_diff 
                weight_g = np.max(diff_g) / max_diff 
                weight_b = np.max(diff_b) / max_diff 
                total_weight = weight_r + weight_g + weight_b
            else:
                total_weight = 1
            diff = cv2.addWeighted(cv2.addWeighted(diff_r, weight_r / total_weight, diff_g, weight_g / total_weight, 0), 
                                   1, diff_b, weight_b / total_weight, 0)
            diff = cv2.convertScaleAbs(diff * 1.5)  # Multiply by scalar to enhance differences

            # Normalize the diff frame to the range [0, 255]
            cv2.normalize(diff, diff, 100, 255, cv2.NORM_MINMAX)
            _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)

            # show thresholded frame for each lane
            # Detect contours based on area
            # Apply edge detection
            edges = cv2.Canny(thresh, 100, 200)
            # Display the edges frame for visualization
            # Draw edges on the frame for visualization
            

            # Find contours on the edges
            contours, _ = cv2.findContours(edges, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1)
            # Draw all contours on the frame for visualization
            # for i, contour in enumerate(contours):
            #     color = (i * 40 % 256, i * 80 % 256, i * 120 % 256)  # Generate a unique color for each contour
            #     cv2.drawContours(viz_frame, [contour], -1, color, 1)
            # Filter contours by area and aspect ratio
            contours = [contour for contour in contours if cv2.boundingRect(contour)[2] <= 100
                                                        and cv2.contourArea(contour) > min_area
                                                        and cv2.contourArea(contour) < max_area
                                                        and cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3] < 8]  # Aspect ratio filter and width filter
            all_contours.append(contours)
        all_contours = [contour for sublist in all_contours for contour in sublist]
        grouped_contours = []
        used = set()
    # GROUP contours
        for i, contour1 in enumerate(all_contours):
            if i in used:
                continue
            group = [contour1]
            (x1, y1), _ = cv2.minEnclosingCircle(contour1)
            for j, contour2 in enumerate(all_contours):
                if i == j or j in used:
                    continue
                (x2, y2), _ = cv2.minEnclosingCircle(contour2)
                if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= avg_lane_width*.9 :
                    group.append(contour2)
                    used.add(j)
            grouped_contours.append(group)
            used.add(i)

        for group in grouped_contours:
            merged = np.vstack(group)
            (x, y), radius = cv2.minEnclosingCircle(merged)
            center_x, center_y = int(x), int(y)
            # Draw a circle at the detected object's center
            # cv2.circle(viz_frame, (center_x, center_y), int(radius), (0, 0, 255), 2)
            # Define fixed-size bounding box centered at detected object
            new_x = max(center_x - bbox_width // 2, 0)
            new_y = max(center_y - bbox_height // 2, 0)
    
            # Ensure bbox does not go out of frame bounds
            new_x = min(new_x, frame.shape[1] - bbox_width)
            new_y = min(new_y, frame.shape[0] - bbox_height)
    
            roi = frame[new_y:new_y + bbox_height, new_x:new_x + bbox_width]
    
            # Save the detected region of interest (ROI)
            cv2.imwrite(f"{output_dir}/{roi_count:04d}.jpg", roi)
    
            # Draw a circle and rectangle on viz_frame
            if radius < 1:
                radius =1 
            cv2.rectangle(viz_frame, (new_x, new_y), (new_x + bbox_width, new_y + bbox_height), (0, 255, 0), 2)
            roi_count += 1

        cv2.imshow("Motion", viz_frame)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF  # Capture key press
        if key == ord('q'):  # Press 'q' to exit the loop
            break
        elif key != 255:  # Any key qpress moves to the next frame
            continue


    cap.release()

cv2.destroyAllWindows()
