#ifndef FEATURES_H
#define FEATURES_H

#include <opencv/cv.h>
#include <opencv/ml.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <numeric>

#include <Python.h>
#include <numpy/arrayobject.h>

class GazeMatrix {
public:
  GazeMatrix();

  void processFrame(cv::Mat frame);

private:
  cv::Mat mOp, mOp2;
  cv::Mat mRvec, mTvec;
  cv::Mat mRvec2, mTvec2;
  cv::Mat camMatrix, camMatrix2;

  PyObject *pProcessEye, *pProcessEyeArgs;
  uint8_t *pyBufData;
  double *pyResData;

  dlib::correlation_tracker tracker;
  dlib::shape_predictor pose_model;
  dlib::frontal_face_detector detector;
  CvRTrees* rtree;

  void _setupIrisDetector();
  void _setupModel();
  void _calculateFeatures(
    cv::Mat& image,
    cv::Rect& area,
    cv::Size& size,
    const std::vector<dlib::rectangle>& faces,
    const std::vector<dlib::full_object_detection>& dets,
    const cv::Scalar color
  );
  cv::Point _findEyeCenter(
    cv::Mat face,
    cv::Rect eye,
    std::string debugWindow,
    double interOcDist
  );
  void _scaleToFastSize(const cv::Mat &src,cv::Mat &dst);
  cv::Point _unscalePoint(cv::Point p, cv::Rect origSize);
  void _loadWithPoints(
    cv::Mat& ip,
    cv::Mat& ip2,
    cv::Mat& img,
    cv::Point leftPupil,
    cv::Point RightPupil,
    std::vector<cv::Point2f> LeftEyePoints,
    std::vector<cv::Point2f> RightEyePoints,
    std::vector<cv::Point2f> LeftEllipse,
    std::vector<cv::Point2f> RightEllipse,
    cv::Point meanLeft,
    cv::Point meanRight,
    double &pitch,
    double &yaw,
    double &roll,
    int &interOcDist
  );
};

#endif
