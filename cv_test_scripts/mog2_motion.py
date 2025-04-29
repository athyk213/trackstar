import cv2
import glob

# Get list of all .MP4 files in the directory
video_files = glob.glob("videos/2-23-resized/*.MP4")

for video_path in video_files:
    print(f"Processing video: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Create a new background subtractor for each video
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=100, detectShadows=True
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for uniformity and faster processing
        frame = cv2.resize(frame, (960, 540))
        
        # Apply background subtraction
        fgMask = backSub.apply(frame)
        
        # Remove shadows: pixels with value 127 are shadows, so set them to 0.
        fgMask[fgMask == 127] = 0
        
        # Optional: Clean up noise using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours of moving regions
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # filter out small regions
                continue
            
            # Get bounding box of the contour and calculate its center
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            
            # Force the bounding box to a fixed size: 75x100
            forced_x = cx - 75 // 2
            forced_y = cy - 100 // 2
            cv2.rectangle(frame, (forced_x, forced_y), (forced_x + 75, forced_y + 100), (0, 255, 0), 2)
        
        # Show the processed frame
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()

cv2.destroyAllWindows()
