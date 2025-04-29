#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <filesystem>


using json = nlohmann::json;
namespace fs = cv::utils::fs;

int main() {
    // Load config file
    std::ifstream configFile("../conf_v2.json");
    if (!configFile.is_open()) {
        std::cerr << "Error: Could not open config file." << std::endl;
        return -1;
    }

    json config;
    configFile >> config;

    // Read parameters from config file
    std::string inputDir = config["paths"]["inputDir"];
    std::string outputDir = config["paths"]["outputDir"];
    int frameSkip = config["frameSkip"];
    int boundingBoxWidth = config["boundingBox"]["width"];
    int boundingBoxHeight = config["boundingBox"]["height"];
    double weight = config["weight"];
    double scalingFactor = config["scalingFactor"];
    int brightnessFactor = config["brightnessFactor"];
    int deltaThresh = config["deltaThresh"];
    int minArea = config["minArea"];
    int topCrop = config["topCrop"];
    int bottomCrop = config["bottomCrop"];

    // Camera intrinsics
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<  
        2.80838381e+03, 0, 1.91024134e+03, 
        0, 2.81144960e+03, 1.07297946e+03, 
        0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.05574808, 0.22661742, -0.00186, 0.00075089, -0.28986981);


    // Start timing
    auto start_total = std::chrono::high_resolution_clock::now();

    // Get all video files from the input directory
    std::vector<cv::String> videoFiles;
    fs::glob(inputDir, "*.MP4", videoFiles);
    int totalFrameCount = 0;
    for (const auto& inputPath : videoFiles) {
        std::string outputPath = outputDir + "/" + inputPath.substr(inputPath.find_last_of("/\\") + 1) + "/";
        if (!fs::exists(outputPath)) {
            fs::createDirectories(outputPath);
        }
        std::cout << "Processing video: " << inputPath << " -> " << outputPath << std::endl;
    
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open " << inputPath << std::endl;
            continue;
        }

        cv::Mat frame, resized_frame, undistortedFrame, grayFrame, blurredFrame, avgFrame, scaledAvgFrame, diffFrame, visualization_frame;
        int frameCounter = 0, roiCount = 0;
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            frameCounter++;
            if (frameCounter % frameSkip != 0) continue;

            frame = frame(cv::Rect(0, topCrop, frameWidth, frameHeight - topCrop - bottomCrop));
            cv::resize(frame, frame, cv::Size(frameWidth/scalingFactor, frameHeight/scalingFactor), cv::INTER_LINEAR);
            cv::undistort(frame, undistortedFrame, cameraMatrix, distCoeffs);
            cv::cvtColor(undistortedFrame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(grayFrame, blurredFrame, cv::Size(3, 3), 0);
            visualization_frame = undistortedFrame.clone();

            if (avgFrame.empty()) {
                grayFrame.convertTo(avgFrame, CV_32F);
            } else {
                cv::accumulateWeighted(grayFrame, avgFrame, weight);
                avgFrame.convertTo(scaledAvgFrame, CV_8U);
                cv::absdiff(grayFrame, scaledAvgFrame, diffFrame);
            }

            cv::Mat threshFrame, dilatedFrame;
            std::vector<std::vector<cv::Point>> contours;

            cv::threshold(diffFrame, threshFrame, deltaThresh, 255, cv::THRESH_BINARY);
            if (!threshFrame.empty()) {
                cv::dilate(threshFrame, dilatedFrame, cv::Mat(), cv::Point(-1, -1), 2);
                cv::findContours(dilatedFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                for (const auto& c : contours) {
                    if (cv::contourArea(c) < minArea) continue;
                    
                    cv::Rect boundingBox = cv::boundingRect(c);
                    int centerX = boundingBox.x + boundingBox.width / 2;
                    int centerY = boundingBox.y + boundingBox.height / 2;
                    int newX = std::max(centerX - boundingBoxWidth / 2, 0);
                    int newY = std::max(centerY - boundingBoxHeight / 2, 0);
                    if (newX + boundingBoxWidth > undistortedFrame.cols || newY + boundingBoxHeight > undistortedFrame.rows) continue;

                    boundingBox = cv::Rect(newX, newY, boundingBoxWidth, boundingBoxHeight);

                    std::string filename = outputPath + "roi_" + std::to_string(roiCount++) + ".jpg";
                    cv::imwrite(filename, undistortedFrame(boundingBox));
                    cv::rectangle(visualization_frame, boundingBox, cv::Scalar(0, 255, 0), 2);
                }
                cv::imshow("Contours Frame", dilatedFrame);
            }
            totalFrameCount += frameCounter;
            cv::imshow("Undistorted Frame", visualization_frame);
            if (cv::waitKey(1) == 'q') break;
        }

        cap.release();
    }
    
    cv::destroyAllWindows();
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "Total time: " << elapsed_total.count() << " seconds to process " << totalFrameCount / frameSkip << " frames." << std::endl;

    return 0;
}
