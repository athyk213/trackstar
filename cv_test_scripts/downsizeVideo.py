import os
import cv2

# Directory containing the videos
input_directory = "videos/2-27"
output_directory = "videos/2-27/downscaled"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".MP4"):  # Process only .mp4 files
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        print(f"Processing {filename}...")

        # Open the video file
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default to 30 FPS if unable to retrieve
        width = 1920
        height = 1080
        out = cv2.VideoWriter(output_path, fourcc, max(1, fps), (width, height))
        # Create VideoWriter object
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 1080p
            resized_frame = cv2.resize(frame, (width, height))
            out.write(resized_frame)

        # Release resources
        cap.release()
        out.release()

print("All videos have been downscaled to 1080p.")