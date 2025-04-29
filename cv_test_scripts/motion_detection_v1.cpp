#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    // Load config.json
    std::ifstream configFile("../conf_v1.json");
    if (!configFile.is_open()) {
        std::cerr << "Error: Could not open config.json." << std::endl;
        return -1;
    }

    json config;
    configFile >> config;

    // Read parameters from config.json
    std::vector<std::pair<std::string, std::string>> paths;
    std::string inputDir = config["paths"]["inputDir"];
    std::string outputDir = config["paths"]["outputDir"];
    std::vector<cv::String> inputFiles;
    cv::utils::fs::glob(inputDir, "*.MP4", inputFiles);
    for (const auto& inputFile : inputFiles) {
        std::string outputFile = outputDir + "/" + inputFile.substr(inputFile.find_last_of("/\\") + 1, inputFile.find_last_of(".") - inputFile.find_last_of("/\\") - 1) + "/";
        paths.push_back({inputFile, outputFile});
    }
    int frameSkip = config["frameSkip"];
    float confidenceThreshold = config["confidenceThreshold"];
    float nmsThreshold = config["nmsThreshold"];
    int boundingBoxWidth = config["boundingBox"]["width"];
    int boundingBoxHeight = config["boundingBox"]["height"];
    float scalingFactor = config["scalingFactor"];
    int topCrop = config["topCrop"];
    int bottomCrop = config["bottomCrop"];
    std::string cfgPath = config["model"]["cfgPath"];
    std::string weightsPath = config["model"]["weightsPath"];

    // Start timing
    auto start_total = std::chrono::high_resolution_clock::now();

    // Load YOLO network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfgPath, weightsPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    for (const auto& pathsPair : paths) {
        auto start = std::chrono::high_resolution_clock::now();
        std::string inputPath = pathsPair.first;
        std::string outputPath = pathsPair.second;

        // Process each video
        std::cout << "Processing video: " << inputPath << " -> " << outputPath << std::endl;
            
        // Open video file
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open the video." << std::endl;
            return -1;
        }

        // Get video properties
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int roiCount = 0;
        cv::Mat frame;
        int frameCount = 0;

        while (true) {
            cap >> frame;
            if (frame.empty()) {
                break;
            }
            frameCount++;
            if (frameCount % frameSkip != 0) {
                continue; // Skip this frame
            }
            
            // Frame preprocessing
            frame = frame(cv::Rect(0, topCrop, frameWidth, frameHeight - topCrop - bottomCrop));
            int frameWidthResized = cvRound(frame.cols / scalingFactor) / 32 * 32;
            int frameHeightResized = cvRound(frame.rows / scalingFactor) / 32 * 32;
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);


            // Prepare input blob for YOLO
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(frameWidthResized, frameHeightResized), cv::Scalar(0, 0, 0), true, false);
            net.setInput(blob);

            // Run forward pass
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            // Process YOLO outputs
            std::vector<int> indices;
            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;

            for (const auto& output : outputs) {
                for (int i = 0; i < output.rows; i++) {
                    float confidence = output.at<float>(i, 4);
                    if (confidence > confidenceThreshold) {
                        cv::Mat scores = output.row(i).colRange(5, output.cols);
                        cv::Point classIdPoint;
                        double maxClassScore;
                        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

                        // Filter for "person" class (class 0 in COCO dataset)
                        if (classIdPoint.x == 0 && maxClassScore > confidenceThreshold) {
                            int centerX = static_cast<int>(output.at<float>(i, 0) * frame.cols);
                            int centerY = static_cast<int>(output.at<float>(i, 1) * frame.rows);
                            int width = static_cast<int>(output.at<float>(i, 2) * frame.cols);
                            int height = static_cast<int>(output.at<float>(i, 3) * frame.rows);

                            // Extend bounding box
                            int x = centerX - boundingBoxWidth / 2;
                            int y = centerY - boundingBoxHeight / 2;

                            boxes.emplace_back(cv::Rect(x, y, boundingBoxWidth, boundingBoxHeight));
                            confidences.push_back(static_cast<float>(confidence));
                        }
                    }
                }
            }

            // Apply Non-Max Suppression
            cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

            // Resize the ROI to a fixed size
            cv::Mat resizedRoi(boundingBoxWidth, boundingBoxHeight, CV_8UC3);

            for (size_t i = 0; i < indices.size(); i++) {
                int idx = indices[i];
                cv::Rect box = boxes[idx];

                // Ensure bounding box fits within the frame
                box &= cv::Rect(0, 0, frame.cols, frame.rows);

                // Extract and save ROI
                cv::Mat roi = frame(box);
                if (roi.size() != cv::Size(boundingBoxWidth, boundingBoxHeight)) {
                    cv::resize(roi, resizedRoi, cv::Size(boundingBoxWidth, boundingBoxHeight));
                } else {
                    resizedRoi = roi;
                }

                // Save ROI to file
                // Create output directory if it does not exist
                cv::String outputDir = outputPath;
                if (!cv::utils::fs::exists(outputDir)) {
                    cv::utils::fs::createDirectories(outputDir);
                }

                std::string filename = outputPath + "roi_" + std::to_string(roiCount++) + ".jpg";
                cv::imwrite(filename, resizedRoi);

                // Draw bounding box
                cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            }

            // Display the frame
            cv::imshow("Person Detection", frame);

            // Exit on pressing 'q'
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken: " << elapsed.count() << " seconds to process " << frameCount << " frames." << std::endl;
    }
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "Time taken: " << elapsed_total.count() << " seconds to process " << paths.size() << " videos." << std::endl;
    return 0;
}
