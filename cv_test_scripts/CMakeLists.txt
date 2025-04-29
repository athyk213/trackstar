cmake_minimum_required(VERSION 3.0)
project(TrackStar)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find nlohmann-json
find_package(nlohmann_json REQUIRED)

# Add executables
add_executable(motion_detection_v1 motion_detection_v1.cpp)
add_executable(motion_detection_v2 motion_detection_v2.cpp)
add_executable(lane_detection lane_detection.cpp)

# Link OpenCV and nlohmann-json to your target
target_link_libraries(motion_detection_v1 ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(motion_detection_v2 ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(lane_detection ${OpenCV_LIBS} nlohmann_json::nlohmann_json)