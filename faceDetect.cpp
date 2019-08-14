#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace cv::dnn;

int main(int argc, char** argv)
{
    const string caffeConfigFile = "./model/deploy.prototxt";
    const string caffeModelFile = "./model/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    const float confidenceThreshold = 0.75f;

    Net net = readNetFromCaffe(caffeConfigFile, caffeModelFile);

    Mat frame = Mat();
    VideoCapture video_capture;

    if (argc == 1)
    {
        printf("Using webcam as video source\n");
        video_capture = VideoCapture(0);
    }
    else if (argc == 2)
    {
        printf("Using %s as video source\n", argv[1]);
        video_capture = VideoCapture(argv[1]);
    }
    else if (argc > 2)
    {
        printf("You can specify path for only one video\n");
        return -1;
    }

    //namedWindow("Video");

    VideoWriter video_writer = VideoWriter("videos/out.mp4", video_capture.get(CAP_PROP_FOURCC), video_capture.get(CAP_PROP_FPS), Size(video_capture.get(CAP_PROP_FRAME_WIDTH), video_capture.get(CAP_PROP_FRAME_HEIGHT)));

    while (video_capture.isOpened())
    {
        milliseconds now = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

        if (video_capture.read(frame) && !frame.empty())
        {

            int frameWidth = frame.size().width;
            int frameHeight = frame.size().height;
            Mat inputBlob = blobFromImage(frame, 1.0, Size(300, 300), mean(frame));
            net.setInput(inputBlob, "data");
            Mat detection = net.forward("detection_out");
            Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
            int i;

            for (i = 0; i < detectionMat.rows; i++)
            {
                int x1, x2, y1, y2;
                float confidence;
                string fileName = "./images/" + to_string(now.count()) + "_" + to_string(i) + ".jpg";
                confidence = detectionMat.at<float>(i, 2);
                if (confidence >= confidenceThreshold)
                {
                    x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
                    y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
                    x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
                    y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
                    Mat face = frame(Rect(Point(x1, y1), Point(x2, y2)));
                    imwrite(fileName, face);
                    rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0));
                    string text = to_string(static_cast<int>(confidence * 100)) + "%";
                    putText(frame, text, Point(x1, y1 - 10), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));
                }
            }
            video_writer.write(frame);
            //imshow("Video", frame);
        }
        else
        {
            break;
        }

        if (waitKey(10) != -1)
            break;
    }
    return 0;
}
