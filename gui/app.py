from flask import Flask, render_template, jsonify, send_from_directory
import os, csv, time

app = Flask(__name__)

# Wait until an infractions directory is created.
cwd = os.getcwd()
infractions_dirs = None
while not infractions_dirs:
    infractions_dirs = [d for d in os.listdir(cwd) 
                        if os.path.isdir(os.path.join(cwd, d)) and d.startswith("infractions")]
    if not infractions_dirs:
        print("Waiting for infractions directories to be created...")
        time.sleep(3)

# Prefer "infractions-live" if available, otherwise "infractions-demo" or the first found
if "infractions-live" in infractions_dirs:
    BASE_DIR = os.path.join(cwd, "infractions-live")
else:
    BASE_DIR = os.path.join(cwd, "infractions-demo") if "infractions-demo" in infractions_dirs else os.path.join(cwd, infractions_dirs[0])

def get_timestamps():
    """Reads timestamps from timestamps.csv."""
    timestamps = []
    timestamps_path = os.path.join(BASE_DIR, 'timestamps.csv')
    try:
        with open(timestamps_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None) # Skip the header
            for row in reader:
                timestamps.append(row[0])  # Assuming timestamp is the first and only column
    except FileNotFoundError:
        print(f"timestamps.csv not found at {timestamps_path}")
    return timestamps

def list_local_images():
    """Fetch image file paths from local storage."""
    athletes = sorted(os.listdir(BASE_DIR))  # List athlete folders
    data = {}
    infraction_counter = 0 # start at 0 to match timestamp index
    timestamps = get_timestamps() # get timestamps

    for athlete in athletes:
        athlete_path = os.path.join(BASE_DIR, athlete)
        if not os.path.isdir(athlete_path):
            continue  # Skip non-folder files

        infractions = sorted(os.listdir(athlete_path), key=lambda x: int(x))
        data[athlete] = {}

        for infraction in infractions:
            infraction_path = os.path.join(athlete_path, infraction)
            if not os.path.isdir(infraction_path):
                continue  # Skip non-folder files

            # Ensure '0.jpg' exists for the top row preview
            if '0.jpg' not in os.listdir(infraction_path):
                continue

            # Get sorted list of frames from -5.jpg to 5.jpg
            frames = sorted(
                [f for f in os.listdir(infraction_path) if f.endswith(".jpg")],
                key=lambda x: int(x.replace(".jpg", ""))
            )

            # Store frames in dictionary
            preview_image = '0.jpg'

            # Store frames in dictionary
            data[athlete][f"Infraction {infraction_counter + 1}"] = {
                "preview": f"/infractions/{athlete}/{infraction}/{preview_image}",
                "frames": [f"/infractions/{athlete}/{infraction}/{frame}" for frame in frames],  # Bottom row sequence
                "timestamp": timestamps[infraction_counter] if infraction_counter < len(timestamps) else "Timestamp not found" # add timestamp to data.
            }
            infraction_counter += 1

    return data

@app.route('/')
def index():
    """Render the index page."""
    return render_template('indexLocalFetching.html')

@app.route('/api/infractions/local')
def get_local_infractions():
    """Returns a JSON list of local image URLs grouped by athlete and infraction."""
    images = list_local_images()
    return jsonify(images)

@app.route('/infractions/<athlete>/<infraction>/<filename>')
def serve_image(athlete, infraction, filename):
    """Serve local images dynamically."""
    image_path = os.path.join(BASE_DIR, athlete, infraction)
    return send_from_directory(image_path, filename)

if __name__ == '__main__':
    app.run(debug=True)