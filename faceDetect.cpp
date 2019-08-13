#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>




int main()
{
    const std::string caffeConfigFile = "./model/deploy.prototxt";
    const std::string caffeModelFile = "./model/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    const float confidenceThreshold = 0.75f;
   
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeModelFile);
   
    cv::Mat frame = cv::Mat();
    cv::VideoCapture video_capture = cv::VideoCapture(1);
    cv::namedWindow("Video");
    
    while(video_capture.isOpened())
    {
        std::chrono::milliseconds now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

        if(video_capture.read(frame) && !frame.empty())
        {
            
            
            int frameWidth = frame.size().width;
            int frameHeight = frame.size().height;
            cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::mean(frame));
            net.setInput(inputBlob, "data");
            cv::Mat detection = net.forward("detection_out");
            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
            int i;
            
            for (i = 0; i < detectionMat.rows; i++)
            {  
                int x1, x2, y1, y2;
                float confidence;
                std::string fileName = "./images/" + std::to_string(now.count()) + "_" + std::to_string(i) + ".jpg"; 
                confidence = detectionMat.at<float>(i, 2);
                if (confidence >= confidenceThreshold)
                {
                    x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
                    y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
                    x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
                    y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
                    cv::Mat face = frame(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
                    cv::imwrite(fileName, face);
                    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));
                    std::string text = std::to_string(static_cast<int>(confidence*100)) + "%";
                    cv::putText(frame, text, cv::Point(x1, y1-10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,0));

                }
            }
            
            cv::imshow("Video", frame);
        }
        
        if (cv::waitKey(10) != -1)
            break;
    }
    return 0;
}

