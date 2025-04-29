#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

int main() {
    cv::VideoCapture cap("../../1920x1080/athy.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    cv::Mat frame, undistortedFrame, warpedFrame, grayFrame, blurredFrame, edgeFrame, avgFrame, scaledAvgFrame, diffFrame;
    int new_width = 960;
    int new_height = 540;

    // Camera calibration parameters
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
        2.80838381e+03, 0, 1.91024134e+03, 
        0, 2.81144960e+03, 1.07297946e+03, 
        0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.05574808, 0.22661742, -0.00186, 0.00075089, -0.28986981);

    // Perspective transformation points
    cv::Point2f srcPts[4] = {
        cv::Point2f(0, 300),  // Top-left
        cv::Point2f(1920, 300), // Top-right
        cv::Point2f(0, 700),  // Bottom-left
        cv::Point2f(1920, 700)  // Bottom-right
    };

    cv::Point2f dstPts[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(new_width, 0),
        cv::Point2f(0, new_height),
        cv::Point2f(new_width, new_height)
    };

    cv::Mat transformMatrix = cv::getPerspectiveTransform(srcPts, dstPts);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Undistort image
        cv::undistort(frame, undistortedFrame, cameraMatrix, distCoeffs);

        // Apply perspective warp
        cv::warpPerspective(undistortedFrame, warpedFrame, transformMatrix, cv::Size(new_width, new_height));

        // Convert to grayscale
        cv::cvtColor(warpedFrame, grayFrame, cv::COLOR_BGR2GRAY);

        // Gaussian blur
        cv::GaussianBlur(grayFrame, blurredFrame, cv::Size(5, 5), 0);

        // Edge detection
        cv::Canny(blurredFrame, edgeFrame, 50, 200);

        // Hough line detection
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edgeFrame, lines, 1, CV_PI / 180, 50, 75, 10);

        // Draw lines on the original frame
        for (size_t i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            double angle = abs(atan2(l[3] - l[1], l[2] - l[0]) * 180 / CV_PI);

            if (angle < 30 || angle > 150) {
                // Draw the line if it is not vertical
                cv::line(warpedFrame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }
        }

        // Show the perspective-warped frame with lane detection
        cv::imshow("Lane Detection", warpedFrame);
        // cv::String outputDir = "../lane_results";
        // if (!cv::utils::fs::exists(outputDir)) {
        //     cv::utils::fs::createDirectories(outputDir);
        // }
        // cv::imwrite(outputDir + "lane.jpg", warpedFrame);
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}