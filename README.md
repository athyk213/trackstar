# Project Trackstar

This repository contains the pipelines for infraction detection in track & field competitions. It uses a custom-trained YOLOv11 classification model paired with computer vision techniques for motion detection. Follow the instructions below to set up the environment and run the different pipelines. This system has been tested with the Autel Evo II Pro drone.

## Data Loading (Optional)
Store any video data in a folder called `videos` in the root directory of this repo.

Example file structure:
```
trackstar/
├── videos/
│   └── 2-08
│       ├── Video-1.mp4
│       ├── Video-2.mp4
│       └── Video-3.mp4
├── CameraCalibration/
├── pipeline/
├── ...
```

---

## Running the Program
Make sure you have Python 3.xx installed on your computer. Then, run the following command in the terminal to install the required dependencies:

```bash
pip3 install -r requirements.txt
```

There are currently 3 different pipelines set up for infraction detection. If you're running either the **Live** or **Full** pipelines, you can view the GUI hosted locally at [http://127.0.0.1:5000/](http://127.0.0.1:5000/) **after** executing the respective command.

### Live Pipeline
This pipeline is used to detect infractions while we are **out on the track**. It uses the `pipeline/live_pipeline.py` file. It will output infractions in a directory called `infractions-live` as well as display them on the GUI. Run the following command to run the live pipeline:

```bash
./run_live.sh
```

### Full Pipeline
This pipeline is used to detect infractions in **videos that are stored in a videos/ directory** (as shown in the example file structure above). It uses the `pipeline/full_pipeline.py` file, where you can modify the input/output directories. Run the following command to run the infraction detection on videos:

```bash
./run_full.sh
```

To kill the processes, press Ctrl+C in the terminal.

### Troubleshooting
There are only a few parameters you will likely need to change in the pipeline files for better **motion detection** performance. These are clearly marked near the top of the scripts. The key parameters are:

- `varThreshold`/`min_area`: `varThreshold` is the threshold for the motion detection algorithm. `min_area` is the minimum area of a bounding box for a detected object. Decreasing either parameter will make it more sensitive to motion and could potentially capture some noise. However, it will also prevent losing frames of the athlete(s) running.
- `visualize`/`visualize_lane_detection`: These are used to visualize the motion detection and lane detection algorithm. If you want to see either of them in action, set this to `True`. NOTE: This will slow down the pipeline.

---

## C++ Motion Detection Algorithms
### Setting Up Required Files
Before you can run the motion detection program, you need to run the following commands to download the required files:
```bash
mkdir cv_test_scripts/yolo
curl -o cv_test_scripts/yolo/yolov4.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
curl -L -o cv_test_scripts/yolo/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```
These commands will download the `yolov4.cfg` and `yolov4.weights` files into the appropriate `cv_test_scripts/yolo` directory.

### Installing Dependencies
Make sure you have OpenCV and nlohmann JSON installed.

**For macOS/Linux**:

- **OpenCV**:
  ```bash
  brew install opencv
  ```
- **nlohmann/json**:
  ```bash
  brew install nlohmann-json
  ```

**For Windows**:
- **OpenCV**:
    ```bash
    .\vcpkg install opencv
    ```
- **nlohmann/json**:
    ```bash
    .\vcpkg install nlohmann-json
    ```

### Building and Running the Motion Detection Program
Once the required files are downloaded, create the build directory, run CMake, and build the project:
```bash
mkdir cv_test_scripts/build && cd cv_test_scripts/build
cmake ..
make
```
After building the project, you can run the motion detection program using either of the following commands:

```bash
./motion_detection_v1
```
or
```bash
./motion_detection_v2
```
`v1`: slower, less finicky, uses YOLOv4 model to detect person running on track
`v2`: faster, more finicky, uses frame differences for motion detection
