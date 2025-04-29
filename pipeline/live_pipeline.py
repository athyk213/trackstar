import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque
from itertools import chain
from horizontalLaneDetection import horizontalLaneDetection, fit_poly
import csv

# User parameters
output_dir = f"infractions-live"
model_path = "models/v3.pt"
frame_skip = 1
bbox_width = 75
bbox_height = 100
sensitivity = 0.5 # if prb of infraction >= sensitivity, it's an infraction

# Parameters for motion detection using background subtractor
history = 500
varThreshold = 50
min_area = 75
top_crop = 100
bottom_crop = 0
cooldown_threshold = 10  # Frames to wait before detecting another infraction
gaussian_kernel = (3, 3)
visualize, visualize_lane_detection = True, False
num_lane_contours = 9  # Number of horizontal lane contours to detect
max_contours = 15

# Load YOLO model and set to eval mode
model = YOLO(model_path)
model.model.eval()

# Open output CSV file for timestamps
os.makedirs(output_dir, exist_ok=True)
csv_file = open(os.path.join(output_dir, "timestamps.csv"), mode="a", newline="", buffering=1)
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp"])

# Process live feed
start_time = time.time()
vid_output_dir = os.path.join(output_dir, "live")
os.makedirs(vid_output_dir, exist_ok=True)
cap = cv2.VideoCapture(0)

# Backgruond Subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=True)
infraction_window = deque(maxlen=5)
frame_buffer = deque(maxlen=10)  # (frame, timestamp, infraction lanes)
active_infraction = False
cooldown_frames = 0
infraction_id = 0
lane_contours = []
previous_avg_width = 0
update_lane_contours = True
infraction_lane_buffer = deque(maxlen=10)  # Stores per-frame lane info from infraction frames
frame_count = 0

while True:
    ret, og_frame = cap.read()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if not ret:
        break
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Preprocess frame: resize and crop
    frame = cv2.resize(og_frame, (960, 540))
    frame = frame[top_crop:frame.shape[0] - bottom_crop, :]
    viz_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, gaussian_kernel, 0)

    ########################################################
    # Motion detection using background subtractor
    fgMask = backSub.apply(frame)
    fgMask = (fgMask == 255).astype('uint8')*255 # Remove shadow pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2) # Clean up noise

    ########################################################
    # Lane detection
    if update_lane_contours:
        lane_contours_new = horizontalLaneDetection(blurred)
        if len(lane_contours_new) != 0:
            avg_width = min([cv2.boundingRect(c)[2] for c in lane_contours_new])
            if len(lane_contours_new) == num_lane_contours and avg_width > .9 * previous_avg_width:
                previous_avg_width = avg_width
                lane_contours = sorted(lane_contours_new, key=lambda c: np.mean(c[:, :, 1]))
    y_values = fit_poly(frame, viz_frame, lane_contours, visualize_lane_detection)

    ########################################################
    # Contour extraction and filtering using fgMask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if len(y_values) >= 2:
        top_c = [y for y in y_values[0] if y is not None]
        bot_c = [y for y in y_values[-1] if y is not None]
        contours = [c for c in contours if min(top_c) <= np.mean(c[:, :, 1]) <= max(bot_c)]
    update_lane_contours = len(contours) == 0
    if len(contours) > max_contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]

    ########################################################
    # Per-frame infraction decision using batched YOLO classification
    frame_has_infraction = False
    frame_infraction_lanes = []
    roi_images = []   # will hold ROI images
    roi_info = []     # will hold center coordinates (cx, cy, bx, by)

    # Build the ROI list from filtered contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        bx = max(cx - bbox_width // 2, 0)
        by = max(cy - bbox_height // 2, 0)
        if bx + bbox_width > frame.shape[1] or by + bbox_height > frame.shape[0]:
            continue
        roi = frame[by:by + bbox_height, bx:bx + bbox_width]
        roi_images.append(roi)
        roi_info.append((cx, cy, bx, by))

    # Run batched inference if there is at least one ROI
    if roi_images:
        results = model(roi_images, verbose=False)  # one inference call for the batch
        for idx, result in enumerate(results):
            pred_class = 0 if result.probs.data[0].item() >= sensitivity else 1  
            cx, cy, bx, by = roi_info[idx]
            if pred_class == 0:  # 0 means infraction
                frame_has_infraction = True
                lane_y_closest = []
                for i, lane_y in enumerate(y_values):
                    if cx < len(lane_y) and lane_y[cx] is not None:
                        lane_y_closest.append((i, lane_y[cx]))
                if lane_y_closest:
                    lane_y_closest = sorted(lane_y_closest, key=lambda x: abs(x[1] - cy))[:2]
                    frame_infraction_lanes.append(num_lane_contours - max(lane_y_closest, key=lambda x: x[0])[0])
            if visualize:
                label = "Infraction" if pred_class == 0 else "No Infraction"
                color = (0, 0, 255) if pred_class == 0 else (0, 255, 0)
                cv2.rectangle(viz_frame, (bx, by), (bx + bbox_width, by + bbox_height), color, 2)
                cv2.putText(viz_frame, label, (bx, max(by - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    ########################################################

    # Update rolling window with per-frame decision
    infraction_window.append(0 if frame_has_infraction else 1)
    if frame_has_infraction:
        infraction_lane_buffer.append(frame_infraction_lanes)
    else:
        frame_infraction_lanes.clear()

    # Combine lane information across frames in the buffer
    infraction_lanes = sorted(list(set(chain.from_iterable(infraction_lane_buffer))))
    frame_buffer.append((og_frame.copy(), now, infraction_lanes))

    # Overlay timestamp and lane info on original frame
    cv2.putText(og_frame, now, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(og_frame, f"Infraction Lane: {infraction_lanes}",
                (og_frame.shape[1] - og_frame.shape[1] // 2, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for i in range(len(frame_buffer)):
        frame_buffer[i] = (frame_buffer[i][0], frame_buffer[i][1], infraction_lanes)

    # Check if rolling window has at least 3 infractions for detection
    detected_infraction = infraction_window.count(0) >= 2

    # If an infraction is detected and not already active, capture buffered frames
    if detected_infraction and not active_infraction and cooldown_frames == 0:
        infraction_id += 1
        infraction_folder = os.path.join(vid_output_dir, f"{infraction_id}")
        os.makedirs(infraction_folder, exist_ok=True)

        # Save last 5 frames from the buffer (-5 to -1)
        buffered_frames = list(frame_buffer)[-6:-1]
        for i, (buffered_frame, ts, lanes) in enumerate(buffered_frames, start=-5):
            frame_name = f"{infraction_folder}/{i}.jpg"
            cv2.putText(buffered_frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(buffered_frame, f"Infraction Lane: {lanes}",
                        (buffered_frame.shape[1] - buffered_frame.shape[1] // 2, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(frame_name, buffered_frame)

        cv2.imwrite(f"{infraction_folder}/0.jpg", og_frame)
        csv_writer.writerow([now])

        active_infraction = True
        post_infraction_frames = 5
        cooldown_frames = cooldown_threshold
        continue

    # Save frames during active infraction period
    if active_infraction:
        frame_name = f"{infraction_folder}/{6 - post_infraction_frames}.jpg"
        cv2.imwrite(frame_name, og_frame)
        post_infraction_frames -= 1
        if post_infraction_frames == 0:
            active_infraction = False
            infraction_window.clear()
            frame_buffer.clear()
            infraction_lane_buffer.clear()

    if cooldown_frames > 0:
        cooldown_frames -= 1

    if visualize:
        cv2.imshow("Frame", viz_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
csv_file.close()
if visualize:
    cv2.destroyAllWindows()