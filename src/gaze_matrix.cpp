// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
 */

#include <opencv/cxcore.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <fstream>
#include <numeric>
#include <opencv/cv.h>
#include <opencv2/features2d/features2D.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <pthread.h>
#include <GLFW/glfw3.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <iostream>
#include <string>
#include <queue>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <tclap/CmdLine.h>

using namespace dlib;
using namespace std;



//Constands for finding eye centers
// Algorithm Parameters
const int kFastEyeWidth = 50;
#define PI 3.14159265
cv::Mat op,op2;
cv::Mat ip, ip2;
cv::Mat L_eyePoint;
cv::Mat R_eyePoint;
cv::Mat axisMatrix;
cv::Mat ModelLeftMatrix;
int interOcDist = -1;
double rot[9] = {0};
double rot2[9] = {0};
cv::Mat backPxls;
std::vector<double> rv(3), tv(3),rv2(3),tv2(3);
cv::Mat rvec(rv),tvec(tv);
cv::Mat rvec2(rv2),tvec2(tv2);
cv::Mat camMatrix, camMatrix2;
cv::Point oldLeftPupil;
cv::Point oldRightPupil;
cv::Point middle;
PyObject *pFunc, *pFunc2, *pArgs, *pArgs2;
uint8_t *pyBufData, *pyBufData2;
double *pyResData, *pyResData2;
cv::RNG rng(12345);
int leftSize = 100, rightSize = 100;
cv::Point RightPupil;
cv::Point leftPupil;
struct sockaddr_un addr;
char buf[1];
int fd,rc;
int counterLeft = 0, counterRight = 0;
string socket_path = "/var/run/rTreeDemo";
string train_path = "/Users/vt/Code/Internal/lab-gaze-tracker-training/std_forrest_train.txt"; //Training file used to create the model
string test_path = "/Users/vt/Code/Internal/lab-gaze-tracker-training/std_forrest_test.txt"; // Test file for the modl
string model_path = "models/normalized_Random_forrest_classifier.yml";
int datamode = 0;
int screenPoint = 0; // Shows Gazepoint when active
int train = 0;
double roll, pitch, yaw;
float magicNumber = 1.0865; //Not really important
CvRTrees* rtree;
float sphereRadius = 13.0382; // Magic number * 12 mm which is the avg radius of eyeball
bool loadModel = false; // Create model vs load model
bool use_camera = false;
string frame_file;

// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 4719
#define ATTRIBUTES_PER_SAMPLE 47
#define NUMBER_OF_TESTING_SAMPLES 195
#define NUMBER_OF_CLASSES 4


// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(string filename, cv::Mat data, cv::Mat classes,
                       int n_samples )
{
  float tmp;

  // if we can't read the input file then return 0
  FILE* f = fopen( filename.c_str(), "r" );
  if( !f )
  {
    printf("ERROR: cannot read file %s\n",  filename.c_str());
    return 0;     // all not OK
  }

  // for each sample in the file

  for(int line = 0; line < n_samples; line++)
  {

    // for each attribute on the line in the file

    for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
    {
      if (attribute < 47)
      {

        // first 43 elements (0-46) in each line are the attributes

        fscanf(f, "%f,", &tmp);
        data.at<float>(line, attribute) = tmp;
      }
      else if (attribute == 47)
      {

        // attribute 47 is the class label {0 ... 9}

        fscanf(f, "%f,", &tmp);
        classes.at<float>(line, 0) = tmp;

      }
    }
  }

  fclose(f);
  return 1;   // all OK
}

// Resuzes Mat object to appropriate size
void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

//Undoes Scaling
cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

//Find's the centre of the Iris from appropriate eye region. This is done by passing the appropriate objects to the Python Iris Finder
cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow, double interOcDist) {
  cv::Mat eyeROIUnscaled = face(eye);
  cv::Mat eyeROI (cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/eye.width) * eye.height),CV_8UC3,pyBufData,cv::Mat::AUTO_STEP);
  scaleToFastSize(eyeROIUnscaled, eyeROI);
  pyResData[0] = (double)eyeROI.rows;
  pyResData[1] = (double)eyeROI.cols;
  pyResData[2] = interOcDist;
  //Iris Finder is called now
  PyObject_CallObject(pFunc, pArgs);
  cv::Point result (pyResData[0],pyResData[1]);
  //Rescale the point
  return unscalePoint(result,eye);
}

//Handles Interfacing with the 3-D model and things such as backproject/ reproject are doen here to get Roll, Pitch, Yaw
void loadWithPoints(cv::Mat& ip, cv::Mat& img,cv::Point leftPupil,cv::Point RightPupil,std::vector<cv::Point2f> LeftEyePoints, std::vector<cv::Point2f> RightEyePoints, std::vector<cv::Point2f> LeftEllipse, std::vector<cv::Point2f> RightEllipse, cv::Point meanLeft, cv::Point meanRight) {
  int max_d = MAX(img.rows,img.cols);
  float delX = 30.0;
  float delY = 30.0;
  float delZ = 30.0;
  int rightCentre, leftCentre;
  std::vector<cv::Point3f> axisPoints, ModelLeft;

  //This will be used to draw the coordinate system on the nose later
  axisPoints.push_back(cv::Point3f(36.8301,78.3185,52.0345)); //nose coordinate
  axisPoints.push_back(cv::Point3f(36.8301 + delX,78.3185,52.0345));  //nose + x displacement
  axisPoints.push_back(cv::Point3f(36.8301,78.3185 + delY,52.0345)); //nose + y displacement
  axisPoints.push_back(cv::Point3f(36.8301,78.3185,52.0345 + delZ)); // nose + z displacement
  axisPoints.push_back(cv::Point3f(70.0602,109.898,20.8234));// r eye
  axisPoints.push_back(cv::Point3f(70.0602 + delX,109.898,20.8234));// r eye + x displacement (v 0)
  axisPoints.push_back(cv::Point3f(70.0602,109.898+delY,20.8234));  // r eye + y displacement(v 0)
  axisPoints.push_back(cv::Point3f(70.0602,109.898,20.8234+delZ));  // r eye + z displacement(v 0)
  axisPoints.push_back(cv::Point3f(2.37427,110.322,21.7776));  // l eye
  axisPoints.push_back(cv::Point3f(2.37427+delX,110.322,21.7776));  // l eye + x displacement(v 314)
  axisPoints.push_back(cv::Point3f(2.37427,110.322+delY,21.7776));  // l eye + y displacement(v 314)
  axisPoints.push_back(cv::Point3f(2.37427,110.322,21.7776+delZ));  // l eye + z displacement(v 314)
  axisPoints.push_back(cv::Point3f(70.0602,109.898,20.8234)); // Eye Centre Right eye
  axisPoints.push_back(cv::Point3f(2.37427,110.322,21.7776)); // Eye Centre Left eye
  axisMatrix = cv::Mat(axisPoints);

  // This was used to try to estimate the pose of the eye for 2-D coordinates currently doesnt work well
  ModelLeft.push_back(cv::Point3f(0,15.5302,17.8295));  // Centre of eye (v 1879)
  ModelLeft.push_back(cv::Point3f(0 +delX,15.5302,17.8295));  // Centre of eye (v 1879)
  ModelLeft.push_back(cv::Point3f(0,15.5302 + delY,17.8295));  // Centre of eye (v 1879)
  ModelLeft.push_back(cv::Point3f(0,15.5302,17.8295 + delZ));  // Centre of eye (v 1879)

  ModelLeftMatrix = cv::Mat(ModelLeft);


  //This is where all the PnP / camera perspective stuff is calculated
  camMatrix = (cv::Mat_<double>(3,3) << max_d, 0, img.cols/2.0,
               0,  max_d, img.rows/2.0,
               0,  0,  1.0);

  double _dc[] = {0,0,0,0};
  cv::solvePnP(op,ip,camMatrix,cv::Mat(1,4,CV_64FC1,_dc),rvec,tvec,false,CV_EPNP);

  cv::Mat rotM(3,3,CV_64FC1,rot);

  //Splits the rotation matrix into a vector as well a matrix
  cv::Rodrigues(rvec,rotM);
  double* _r = rotM.ptr<double>();
  // printf("rotation mat: \n %.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",
  //   _r[0],_r[1],_r[2],_r[3],_r[4],_r[5],_r[6],_r[7],_r[8]);
  // printf("trans vec: \n %.3f %.3f %.3f\n",tv[0],tv[1],tv[2]);

  double _pm[12] = {_r[0],_r[1],_r[2],tv[0],
                    _r[3],_r[4],_r[5],tv[1],
                    _r[6],_r[7],_r[8],tv[2]};

  cv::Matx34d P(_pm);
  cv::Mat KP = camMatrix * cv::Mat(P);

  // EYE MODEL PnP
  camMatrix2 = (cv::Mat_<double>(3,3) << max_d, 0, img.cols/2.0,
                0,  max_d, img.rows/2.0,
                0,  0,  1.0);

  double _dc2[] = {0,0,0,0};
  cv::solvePnP(op2,ip2,camMatrix2,cv::Mat(1,4,CV_64FC1,_dc),rvec2,tvec2,false,CV_EPNP);

  cv::Mat rotM2(3,3,CV_64FC1,rot2);
  cv::Rodrigues(rvec2,rotM2);
  double* _r2 = rotM2.ptr<double>();
  double _pm2[12] = {_r2[0],_r2[1],_r2[2],tv2[0],
                     _r2[3],_r2[4],_r2[5],tv2[1],
                     _r2[6],_r2[7],_r2[8],tv2[2]};

  cv::Matx34d P2(_pm2);
  cv::Mat KP2 = camMatrix2 * cv::Mat(P2);

  cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
  cv::Point2f nose, lEye,rEye,ModelLeftEye;
  cv::RotatedRect lEllipse, rEllipse;

  //Create an ellipse for the points in the left eye contour
  if(LeftEllipse.size() >= 6) {
    lEllipse = fitEllipse(LeftEllipse);
  }

  //Create an ellipse for the points in the right eye contour
  if(RightEllipse.size() >= 6) {
    rEllipse = fitEllipse(RightEllipse);
  }

  // The pose for the nose is drawn here
  for (int i=0; i<axisMatrix.rows; i++) {
    cv::Mat_<double> X = (cv::Mat_<double>(4,1) << axisMatrix.at<float>(i,0),axisMatrix.at<float>(i,1),axisMatrix.at<float>(i,2),1.0);
    cv::Mat_<double> opt_p = KP * X;
    cv::Point2f opt_p_img(opt_p(0)/opt_p(2),opt_p(1)/opt_p(2));

    if(i == 0)
      nose = opt_p_img;

    else if(i == 1) {
      if(nose == opt_p_img)
        cv::circle(img, opt_p_img, 5, cv::Scalar(255,0,255), 5);

      cv::line(img, nose, opt_p_img, cv::Scalar(255,0,0),5);
    }
    else if(i == 2) {
      cv::line(img, nose, opt_p_img, cv::Scalar(0,255,0),5);
    }
    else if(i == 3) {
      cv::line(img, nose, opt_p_img, cv::Scalar(0,0,255),5);
    }

    else if(i == 4) {
      rEye = opt_p_img;
      cv::LineIterator it2(img, opt_p_img, RightPupil, 8);
      rightCentre = opt_p_img.x;
      int counter = 0;
      while(counter < 30) {
        it2++;
        counter++;
      }
    }

    else if(i == 5) {

      // line(img, rEye, opt_p_img, cv::Scalar(255,0,0),5);
    }
    else if(i == 6) {
      // line(img, rEye, opt_p_img, cv::Scalar(0,255,0),5);
    }
    else if(i == 7) {
      // line(img, RightPupil, opt_p_img, cv::Scalar(0,0,255),5);
    }

    else if(i == 8) {
      lEye = opt_p_img;
      // circle(img, opt_p_img, 3, Scalar(255,0,0), 2);
      cv::LineIterator it(img, opt_p_img, leftPupil, 8);
      leftCentre = opt_p_img.x;
      int counter = 0;
      while(counter < 30) {
        it++;
        counter++;
      }

      // line(img, opt_p_img, it.pos(), cv::Scalar(0,0,255),5);
    }
    else if(i == 9) {

      // line(img, lEye, opt_p_img, cv::Scalar(255,0,0),5);
    }
    else if(i == 10) {
      // line(img, lEye, opt_p_img, cv::Scalar(0,255,0),5);
    }
    else if(i == 11) {
      // line(img, leftPupil, opt_p_img, cv::Scalar(0,0,255),5);
    }
    else if(i == 12) {
      cv::LineIterator it(img, opt_p_img, leftPupil, 8);
      leftCentre = opt_p_img.x;
      int counter = 0;
      while(counter < 30) {
        it++;
        counter++;
      }
      cv::circle(img, opt_p_img, 2, cv::Scalar(0,255,0), 2);
    }
    else if(i == 13) {
      cv::LineIterator it(img, opt_p_img, RightPupil, 8);
      leftCentre = opt_p_img.x;
      int counter = 0;
      while(counter < 30) {
        it++;
        counter++;
      }
      cv::circle(img, opt_p_img, 2, cv::Scalar(0,255,0), 2);
    }
  }

  //Failed eye Pose
  for (int i=0; i<ModelLeftMatrix.rows; i++) {
    cv::Mat_<double> X2 = (cv::Mat_<double>(4,1) << ModelLeftMatrix.at<float>(i,0),ModelLeftMatrix.at<float>(i,1),ModelLeftMatrix.at<float>(i,2),1.0);
    cv::Mat_<double> opt_p2 = KP2 * X2;
    cv::Point2f opt_p_img2(opt_p2(0)/opt_p2(2),opt_p2(1)/opt_p2(2));

    if(i == 0)
      ModelLeftEye = opt_p_img2;

    else if(i == 1) {
      // cv::line(img, ModelLeftEye, opt_p_img2, cv::Scalar(255,0,0),5);
    }
    else if(i == 2) {
      // cv::line(img, ModelLeftEye, opt_p_img2, cv::Scalar(0,255,0),5);
    }
    else if(i == 3) {
      // cv::line(img, ModelLeftEye, opt_p_img2, cv::Scalar(0,0,255),5);
    }
  }

  //Inter Ocular Distance that is later fed into the Python code
  interOcDist = abs(leftCentre - rightCentre);
  // cout<<"Inter occular Distance: "<<interOcDist<<endl;
  cv::Point2f leftLeftEdge, leftRightEdge, rightLeftEdge, rightRightEdge; // convention is lower case denotes which eye , upper case denotes left or right edge
  //reproject object points - check validity of found projection matrix
  for (int i=0; i<op.rows; i++) {
    cv::Mat_<double> X = (cv::Mat_<double>(4,1) << op.at<float>(i,0),op.at<float>(i,1),op.at<float>(i,2),1.0);
    cv::Mat_<double> opt_p = KP * X;
    cv::Point2f opt_p_img(opt_p(0)/opt_p(2),opt_p(1)/opt_p(2));

    if(i == 1) {
      leftLeftEdge.x = opt_p_img.x;
      leftLeftEdge.y = opt_p_img.y;
    }
    else if( i == 2) {
      leftRightEdge.x = opt_p_img.x;
      leftRightEdge.y = opt_p_img.y;
    }
    else if( i == 3) {
      rightLeftEdge.x = opt_p_img.x;
      rightLeftEdge.y = opt_p_img.y;
    }
    else if( i == 4) {
      rightRightEdge.x = opt_p_img.x;
      rightRightEdge.y = opt_p_img.y;
    }
  }

  double x, ret, val;
  val = 180.0 / PI;

  //Standard roll pitch yaw calculations from rotation matrix
  yaw = atan2 (_r[3], _r[0]) * val;
  pitch = atan2(-1*_r[6],sqrt(_r[7]*_r[7] + _r[8]*_r[8])) * val;
  roll = atan(_r[7]/_r[8]) * val;

  //These will later be used to draw points on the screen to show each respective quadrant
  cv::Point TopLeft, TopRight, BotLeft, BotRight;
  TopLeft.x = img.cols/4;
  TopLeft.y = img.rows/4;
  BotLeft.x = img.cols/4;
  BotLeft.y = 3*img.rows/4;
  TopRight.x = 3*img.cols/4;
  TopRight.y = img.rows/4;
  BotRight.x = 3*img.cols/4;
  BotRight.y = 3*img.rows/4;

  float LeftEyeHeight = (LeftEyePoints[0].y + LeftEyePoints[2].y)/ 2;
  float RightEyeHeight = (RightEyePoints[0].y + RightEyePoints[2].y)/2;
  float LeftEyeCornerAvg = (leftLeftEdge.y + leftRightEdge.y)/2;

  leftSize = int(leftRightEdge.x - leftLeftEdge.x);
  rightSize = int(rightRightEdge.x - rightLeftEdge.x);
  int xDiff = leftPupil.x - lEllipse.center.x;

  int prob;

  float yDiff = abs(leftPupil.y - lEllipse.center.y);
  cv::Point2f gazePoint;
  rotM = rotM.t();// transpose to conform with majorness of opengl matrix

}

/*This function does the following :
   1. Draws Facial Landmarks, Iris Centre and Head-Pose Vectors
   2. Calculates the Standard Deviation and Mean Face point of all 68 points
   3. Crop's out Left and Right Eye Region for Iris Detection
   4. Prepares the Gaze Matrix that will be used at runtime for Gaze Screen Location Prediction. Press 'u' during runtime to activate prediction output or 'i' to stop.
   5. Gathers training/ test data during run time commands include:
      'q' Top Left corner is now recording data so look at top left for good training samples
      'w' Top Right corner is now recording data so look at top right for good training samples
      'a' Bot Left corner is now recording data so look at bot left for good training samples
      's' Bot Right corner is now recording data so look at bot right for good training samples
      'r' Stop recording data
      't' Outputs to Test file instead of training, all the same above commands still apply
   6. Final Gaze Location
 */

void render_face_detections (cv::Mat& image, cv::Rect& area, cv::Size& size, const std::vector<rectangle>& trackers, const std::vector<rectangle>& faces, const std::vector<full_object_detection>& dets, const cv::Scalar color)
{
  // Variables
  cv::Mat newImg, src_gray,crop,crop2, eyeMap,leftCrop, rightCrop;
  newImg= image.clone();
  std::vector<cv::Point2f> srcImagePoints, LeftEyePoints, RightEyePoints, LeftEyePointsModel,LeftEllipse, RightEllipse, allKeyPoints;
  std::vector<float> allKeyPointsX, allKeyPointsY;
  cv::Point LeftGaze, RightGaze, RightCorner, LeftCorner, TopCorner, BotCorner, p1, meanLeft, meanRight;
  cv::Rect LeftEye, RightEye, rectLeft, rectRight;
  cv::Point TopLeft, TopRight, BotLeft, BotRight;
  TopLeft.x = image.cols/4;
  TopLeft.y = image.rows/4;
  BotLeft.x = image.cols/4;
  BotLeft.y = 3*image.rows/4;
  TopRight.x = 3*image.cols/4;
  TopRight.y = image.rows/4;
  BotRight.x = 3*image.cols/4;
  BotRight.y = 3*image.rows/4;

  const long points[] = {1,16,0,28,30,0,18,21,0,23,26,0,31,35,1,37,41,1,43,47,1,49,59,1,61,67,1,-1};
  for (unsigned long i = 0; i < trackers.size(); ++i)
    cv::rectangle(image, 2*cv::Point(trackers[i].left(),trackers[i].top()),2*cv::Point(trackers[i].right(),trackers[i].bottom()), CV_RGB(0,0,255), 1, CV_AA);

  float xScale = ((float)area.width) / size.width, yScale = ((float)area.height) / size.height;

  for (unsigned long i = 0; i < faces.size(); ++i)
    cv::rectangle(image, cv::Point_<float>(faces[i].left()*xScale+area.x,faces[i].top()*yScale+area.y),cv::Point_<float>(faces[i].right()*xScale+area.x,faces[i].bottom()*yScale+area.y), CV_RGB(255,0,0), 1, CV_AA);

  //Create Model
  for (unsigned long i = 0; i < dets.size(); ++i)
  {
    DLIB_CASSERT(dets[i].num_parts() == 68,
                 "\t std::vector<image_window::overlay_line> render_face_detections()"
                 << "\n\t Invalid inputs were given to this function. "
                 << "\n\t dets["<<i<<"].num_parts():  " << dets[i].num_parts()
                 );

    const full_object_detection& d = dets[i];

    /************************ DRAW POINTS AND LINES ********************************/

    for (int p = 0; points[p] > 0; p += 3)
    {
      for (unsigned long i = points[p]; i <= points[p+1]; i++)
        cv::line(image,cv::Point_<float>(d.part(i).x()*xScale+area.x,d.part(i).y()*yScale+area.y),
                 cv::Point_<float>(d.part(i-1).x()*xScale+area.x,d.part(i-1).y()*yScale+area.y), color, 1, CV_AA);
      if (points[p+2])
        cv::line(image,cv::Point_<float>(d.part(points[p]-1).x()*xScale+area.x,d.part(points[p]-1).y()*yScale+area.y),
                 cv::Point_<float>(d.part(points[p+1]).x()*xScale+area.x,d.part(points[p+1]).y()*yScale+area.y), color, 1, CV_AA);
    }

    for(unsigned long i =0; i<=67; i++)
    {
      p1.x = d.part(i).x()*xScale+area.x;
      p1.y = d.part(i).y()*yScale+area.y;
      std::string x = to_string(i);
      allKeyPoints.push_back(p1);
      allKeyPointsX.push_back(p1.x);
      allKeyPointsY.push_back(p1.y);

      //Push the eye contour points back intp vectors
      if(i == 36)
        LeftEllipse.push_back(p1);

      if(i == 37)
        LeftEllipse.push_back(p1);

      if(i == 38)
        LeftEllipse.push_back(p1);

      if(i == 39)
        LeftEllipse.push_back(p1);

      if(i == 40)
        LeftEllipse.push_back(p1);

      if(i == 41)
        LeftEllipse.push_back(p1);

      if(i == 42)
        RightEllipse.push_back(p1);

      if(i == 43)
        RightEllipse.push_back(p1);

      if(i == 44)
        RightEllipse.push_back(p1);

      if(i == 45)
        RightEllipse.push_back(p1);

      if(i == 46)
        RightEllipse.push_back(p1);

      if(i == 47)
        RightEllipse.push_back(p1);

      //  cv::putText(image,x,p1, CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
      if(i == 36 || i == 39 || i == 42 || i == 45 || i == 30 || i == 48 || i == 54)
      {
        srcImagePoints.push_back(cv::Point(d.part(i).x()*xScale+area.x,d.part(i).y()*yScale+area.y));
        if(i == 36)
        {
          LeftEyePoints.push_back(p1);
          LeftEye.x = p1.x - 20*leftSize/100;
          LeftEye.y = p1.y - 50*leftSize/100;
          LeftEye.width = leftSize;
          LeftEye.height = leftSize;

          //This prevents the program from crashing if a face is lost
          try
          {
            crop = newImg(LeftEye);
            leftPupil = findEyeCenter(newImg,LeftEye,"Left Eye",interOcDist);

          }
          catch(...)
          {
            cout<<"LEFT BOUNDING BOX EXCEPTION"<<endl;
          }

          leftPupil.x += LeftEye.x;
          leftPupil.y += LeftEye.y;

        }

        if(i == 39)
        {
          LeftEyePoints.push_back(p1);
        }


        if(i == 42) {
          RightEyePoints.push_back(p1);
          RightEye.x = p1.x - 5*rightSize/100;
          RightEye.y = p1.y - 50*rightSize/100;
          RightEye.width = rightSize;
          RightEye.height = rightSize;

          try{
            crop2 = newImg(RightEye);
            RightPupil = findEyeCenter(newImg,RightEye,"Right Eye",interOcDist);
          }
          catch(...) {
            cout<<"RIGHT EYE EXCEPTION HERE !!!!!!!!!!!!!!!!!!!!"<<endl;
          }
          RightPupil.x += RightEye.x;
          RightPupil.y += RightEye.y;
          RightEyePoints.push_back(RightPupil);
        }

        if (i == 45) {
          RightEyePoints.push_back(p1);
        }

      }
    }
  }

  //Calculate the mean and standard deviation of the face from all 68 points
  double sumX = std::accumulate(allKeyPointsX.begin(), allKeyPointsX.end(), 0.0);
  double meanX = sumX / allKeyPointsX.size();

  double sq_sumX = std::inner_product(allKeyPointsX.begin(), allKeyPointsX.end(), allKeyPointsX.begin(), 0.0);
  double stdevX = std::sqrt(sq_sumX / allKeyPointsX.size() - meanX * meanX);


  double sumY = std::accumulate(allKeyPointsY.begin(), allKeyPointsY.end(), 0.0);
  double meanY = sumY / allKeyPointsY.size();

  double sq_sumY = std::inner_product(allKeyPointsY.begin(), allKeyPointsY.end(), allKeyPointsY.begin(), 0.0);
  double stdevY = std::sqrt(sq_sumY / allKeyPointsY.size() - meanY * meanY);

  cv::RotatedRect lEllipse, rEllipse;
  float startX = 10000, startY = 10000;
  float right_startX = 10000, right_startY = 10000;

  if(LeftEllipse.size() >= 6) {
    for(unsigned long i = 0; i < LeftEyePoints.size(); i++)
    {
      if(LeftEyePoints[i].x < startX)
        startX = LeftEyePoints[i].x;
      if(LeftEyePoints[i].y < startY)
        startY = LeftEyePoints[i].y;
    }

    //Hard coded region for cropping eye regions
    rectLeft.x = startX - 10*leftSize/100;
    rectLeft.y = startY - 20*leftSize/100;
    lEllipse = fitEllipse(LeftEllipse);
    rectLeft.width = lEllipse.size.height + 10*leftSize/100;
    rectLeft.height = 3*lEllipse.size.height/4;

    try
    {
      leftCrop = newImg(rectLeft);
      leftPupil = findEyeCenter(newImg,rectLeft,"Left Eye",interOcDist);
      leftPupil.x += rectLeft.x;
      leftPupil.y += rectLeft.y;
      RightCorner.x = leftPupil.x - 15*leftSize/100;
      RightCorner.y = leftPupil.y;
      LeftCorner.x = leftPupil.x + 15*leftSize/100;
      LeftCorner.y = leftPupil.y;
      TopCorner.x = leftPupil.x;
      TopCorner.y = leftPupil.y - 15*leftSize/100;
      BotCorner.x = leftPupil.x;
      BotCorner.y = leftPupil.y + 15*leftSize/100;
      LeftEyePointsModel.push_back(leftPupil);
      LeftEyePointsModel.push_back(RightCorner);
      LeftEyePointsModel.push_back(LeftCorner);
      LeftEyePointsModel.push_back(TopCorner);
      LeftEyePointsModel.push_back(BotCorner);
      circle(image,leftPupil,2, cv::Scalar(255,125,255), 2);

    }
    catch(...)
    {
      cout<<"LEFT BOUNDING BOX EXCEPTION"<<endl;
    }
  }

  if(RightEllipse.size() >= 6) {
    for(unsigned long i = 0; i < RightEyePoints.size(); i++)
    {
      if(RightEyePoints[i].x < right_startX)
        right_startX = RightEyePoints[i].x;
      if(RightEyePoints[i].y < right_startY)
        right_startY = RightEyePoints[i].y;
    }

    rectRight.x = right_startX - 10*rightSize/100;
    rectRight.y = right_startY - 20*rightSize/100;
    rEllipse = fitEllipse(RightEllipse);
    rectRight.width = rEllipse.size.height + 10*rightSize/100;
    rectRight.height = 3*rEllipse.size.height/4;

    try
    {
      rightCrop = newImg(rectRight);
      RightPupil = findEyeCenter(newImg,rectRight,"Left Eye",interOcDist);
      RightPupil.x += rectRight.x;
      RightPupil.y += rectRight.y;
      circle(image,RightPupil,2, cv::Scalar(0,0,255), 2);

    }
    catch(...)
    {
      cout<<"Right BOUNDING BOX EXCEPTION"<<endl;
    }
  }
  ip = cv::Mat(srcImagePoints);
  ip2 = cv::Mat(LeftEyePointsModel);

  if(LeftEllipse.size() == 6) {
    cv::Mat mean_;
    cv::reduce(LeftEllipse, mean_, CV_REDUCE_AVG, 1);
    // convert from Mat to Point - there may be even a simpler conversion,
    // but I do not know about it.
    cv::Point2f mean(mean_.at<float>(0,0), mean_.at<float>(0,1));
    meanLeft.x = mean.x;
    meanLeft.y = mean.y;
  }

  if(RightEllipse.size() == 6) {
    cv::Mat mean_;
    cv::reduce(RightEllipse, mean_, CV_REDUCE_AVG, 1);
    cv::Point2f mean(mean_.at<float>(0,0), mean_.at<float>(0,1));
    meanRight.x = mean.x;
    meanRight.y = mean.y;

  }

  cv::Point leftCentre, rightCentre, leftDisplacement, rightDisplacement;
  leftCentre.x = (LeftEyePoints[0].x + LeftEyePoints[2].x)/2;
  leftCentre.y = (LeftEyePoints[0].y + LeftEyePoints[2].y)/2;
  leftDisplacement.x = leftPupil.x - lEllipse.center.x;
  leftDisplacement.y = leftPupil.y - lEllipse.center.y;
  rightDisplacement.x = RightPupil.x - rEllipse.center.x;
  rightDisplacement.y = RightPupil.y - rEllipse.center.y;
  std::ofstream trainForrest, trainForrest2;
  cv::Mat var_matrix;
  cv::Mat gazeMatrix= cv::Mat(1, ATTRIBUTES_PER_SAMPLE, CV_32FC1); // this is what we will use at runtime to predict
  double result = -1;

  // Create the Gaze Matrix
  if(LeftEllipse.size() == 6 && RightEllipse.size() == 6) {
    int attribute_counter = 0;
    for (std::vector<cv::Point2f>::iterator i = LeftEllipse.begin(); i != LeftEllipse.end(); ++i)
    {

      gazeMatrix.at<float>(0, attribute_counter) = (i->x - meanX)/stdevX;
      attribute_counter++;
      gazeMatrix.at<float>(0, attribute_counter) = (i->y - meanY)/stdevY;
      attribute_counter++;
    }

    for (std::vector<cv::Point2f>::iterator i = RightEllipse.begin(); i != RightEllipse.end(); ++i)
    {
      gazeMatrix.at<float>(0, attribute_counter) = (i->x - meanX)/stdevX;
      attribute_counter++;
      gazeMatrix.at<float>(0, attribute_counter) = (i->y - meanY)/stdevY;
      attribute_counter++;
    }

    gazeMatrix.at<float>(0, attribute_counter) = roll;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = pitch;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = yaw;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (meanLeft.x - meanX)/stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (meanLeft.y - meanY)/stdevY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (leftDisplacement.x)/stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (leftDisplacement.y)/stdevY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (rightDisplacement.x)/stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (rightDisplacement.y)/stdevY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (leftPupil.x - meanX)/stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (leftPupil.y - meanY)/stdevY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (lEllipse.center.x - meanX)/stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (lEllipse.center.y - meanY)/stdevY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (meanRight.x - meanX)/stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (meanRight.y - meanY)/stdevY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (RightPupil.x - meanX)/ stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (RightPupil.y - meanY)/ stdevY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (rEllipse.center.x - meanX)/ stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = (rEllipse.center.y - meanY)/ stdevY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = meanX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = meanY;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = stdevX;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = stdevY;
    attribute_counter++;
    result = rtree->predict(gazeMatrix, cv::Mat());

    cout << "gaze matrix: " << gazeMatrix << endl;
  }

  if(result >= 0) {
    cout << "result: " << result << endl;
  }

  if(srcImagePoints.size() == 7 && LeftEyePointsModel.size() == 5) {
    loadWithPoints(ip,image,leftPupil,RightPupil,LeftEyePoints,RightEyePoints, LeftEllipse, RightEllipse, meanLeft, meanRight);
  }

  rightCentre.x = (RightEyePoints[0].x + RightEyePoints[2].x)/2;
  rightCentre.y = (RightEyePoints[0].y + RightEyePoints[2].y)/2;
  srcImagePoints.clear();
  LeftEyePointsModel.clear();
  LeftEllipse.clear();
  LeftEyePoints.clear();
  RightEyePoints.clear();

}


pthread_mutex_t _frameMut, *frameMut=&_frameMut;
cv::Mat *sharePtr = NULL;
int userInput = 0;

// Main Function
int main(int argc, const char **argv)
{
  /*****************************************************************************/

  // Wrap everything in a try block.  Do this every time,
  // because exceptions will be thrown for problems.
  try {
    TCLAP::CmdLine cmd("Gaze Tracker", ' ', "0.1");
    TCLAP::UnlabeledValueArg<string> pictureArg("file","Picture file to calculate on", false, "", "string", cmd);
    TCLAP::SwitchArg loadModelSwitch("l","load-model","Loads model instead of training. -l, -r, -e must all be specified together to load model", cmd, false);
    TCLAP::SwitchArg useCameraSwitch("c","use-camera","Loads picture file from camera", cmd, false);
    TCLAP::ValueArg<string> trainArg("r","train-file","File to train on.",false,train_path,"string", cmd);
    TCLAP::ValueArg<string> testArg("e","test-file","File to test on",false,test_path,"string", cmd);
    TCLAP::ValueArg<string> modelArg("m","model-file","Model file to use",false,model_path,"string", cmd);

    // Parse the argv array.
    cmd.parse( argc, argv );

    loadModel = loadModelSwitch.getValue();
    train_path = trainArg.getValue();
    test_path = testArg.getValue();
    model_path = modelArg.getValue();
    use_camera = useCameraSwitch.getValue();

    if(!use_camera) {
      frame_file = pictureArg.getValue();
      cout << "frame file: " << frame_file;
    }
  }
  catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }

  pthread_t _glThread, *glThread = &_glThread;
  pthread_attr_t _tAttr, *tAttr = &_tAttr;

  int mode = 0;
  char text[256];
  /* Modes:
     1. Face tracking - track tight area (scale so face is 100x100 (?))
     2. Lost face for a frame - detect over looser area
     3. Lost face for 2 frames - detect over looser area
     ...
     0. No face detected - detect over entire frame (5fps)
   */

  GLFWwindow* window;

  char syspath[100], pwd[100];

  PyObject *pName, *pName2, *pModule, *pModule2, *pIn, *pIn2, *pOut, *pOut2;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  strcpy(syspath, "sys.path.append('");
  if (getcwd(pwd, sizeof(pwd)) != NULL)
    strcat(syspath, pwd);
  strcat(syspath, "/models')");
  PyRun_SimpleString(syspath);
  //cout<<syspath<<endl;
  pName = PyString_FromString("IrisFinder");
  pModule = PyImport_Import(pName);
  if (pModule == NULL)
  {
    PyErr_Print();
    printf ("Couldn't load iris finder module\n");
    exit(1);
  }

  pIn = PyObject_GetAttrString(pModule, "cvBuffer");
  pOut = PyObject_GetAttrString(pModule, "resBuffer");
  pFunc = PyObject_GetAttrString(pModule, "processEye");
  pArgs = PyTuple_New(0);

  if (pIn == NULL || pOut == NULL || pFunc == NULL)
  {

    printf ("Iris finder module not valid\n");
    exit(1);
  }

  pyBufData = (uint8_t *)PyArray_DATA(pIn);
  pyResData = (double *)PyArray_DATA(pOut);

/*  [>********************* Unix Socket Initialization START ***************************<]*/

  //if ( (fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
  //perror("socket error");
  //exit(-1);
  //}

  //memset(&addr, 0, sizeof(addr));
  //addr.sun_family = AF_UNIX;
  //strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path)-1);

  //if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
  //perror("connect error");
  //exit(-1);
  /*}*/
/********************** Unix Socket Initialization END ****************************/

  std::vector<cv::Point3f> modelPoints, EyeModel;
  //new model points:

  EyeModel.push_back(cv::Point3f(0,15.5302,17.8295));  // Centre of eye (v 1879)
  EyeModel.push_back(cv::Point3f(6.52994,15.5302,15.3021));  //Right Eye corner
  EyeModel.push_back(cv::Point3f(-6.52994, 15.5302, 15.302));  // Left Corner (v 1879)
  EyeModel.push_back(cv::Point3f(0, 22.0601, 15.302));  // Top Corner (v 1879)
  EyeModel.push_back(cv::Point3f(0, 9.0023, 15.302));  // Bot Corner (v 1879)
  //FEyeModel.push_back(cv::Point3f(0,15.5302/2,17.8295/2));


  // modelPoints.push_back(Point3f(70.0602,109.898,20.8234)); // Eye Centre Right eye
  // modelPoints.push_back(Point3f(2.37427,110.322,21.7776)); // Eye Centre Left eye
  modelPoints.push_back(cv::Point3f(36.8301,78.3185,52.0345));  //nose (v 1879)
  modelPoints.push_back(cv::Point3f(-12.6448,108.891,11.8577));// Left edge Left eye
  modelPoints.push_back(cv::Point3f(18.6566,106.811,18.9713)); // Right edge Left eye
  modelPoints.push_back(cv::Point3f(54.0573,106.811,18.4686)); // Left edge Right eye
  modelPoints.push_back(cv::Point3f(85.1435,108.891,10.4489)); // Right edge RIght eye
  modelPoints.push_back(cv::Point3f(14.8498,51.0115,30.2378));  // l mouth (v 1502)
  modelPoints.push_back(cv::Point3f(58.1825,51.0115,29.6224));  // r mouth (v 695)

  op = cv::Mat(modelPoints);
  op2 = cv::Mat(EyeModel);

  rvec = cv::Mat(rv);
  rvec2 = cv::Mat(rv2);

  double _d[9] = {1,0,0,
                  0,-1,0,
                  0,0,-1};
  double _d2[9] = {1,0,0,
                   0,-1,0,
                   0,0,-1};

  Rodrigues(cv::Mat(3,3,CV_64FC1,_d),rvec);
  Rodrigues(cv::Mat(3,3,CV_64FC1,_d2),rvec2);

  tv[0]=0; tv[1]=0; tv[2]=1;
  tvec = cv::Mat(tv);

  tv2[0]=0; tv2[1]=0; tv2[2]=1;
  tvec2 = cv::Mat(tv2);

  camMatrix = cv::Mat(3,3,CV_64FC1);
  camMatrix2 = cv::Mat(3,3,CV_64FC1);

  try
  {
    if (!glfwInit())
      exit(EXIT_FAILURE);

    window = glfwCreateWindow(1920, 1080, "Face tracker", NULL, NULL);
    if (!window)
    {
      glfwTerminate();
      exit(EXIT_FAILURE);
    }

    pthread_attr_init(tAttr);
    pthread_attr_setdetachstate (tAttr, PTHREAD_CREATE_JOINABLE);

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("models/shape_predictor_68_face_landmarks.dat") >> pose_model;
    correlation_tracker tracker;

    // Grab and process frames until the main window is closed by the user.
    int64 t1, t0 = cvGetTickCount();
    int fnum=0;
    double fps=0;
    cv::Mat shared;
    sharePtr = &shared;

    cv::Mat frame, small, track;

    // acquire image
    if(use_camera) {
      cv::VideoCapture cap(0);
      cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
      cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
      cap >> frame;
      cv::flip(frame, frame, 1);
    }
    else {
      frame = cv::imread(frame_file, 1);
    }


    cv_image<bgr_pixel> cframe(frame);

    std::vector<rectangle> tracked;

    /***************** RANDOM TREES ****************/
    // lets just check the version first

    printf ("OpenCV version %s (%d.%d.%d)\n",
            CV_VERSION,
            CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);

    // define training data storage matrices (one for attribute examples, one
    // for classifications)

    cv::Mat training_data = cv::Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    cv::Mat training_classifications = cv::Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

    //define testing data storage matrices

    cv::Mat testing_data = cv::Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    cv::Mat testing_classifications = cv::Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

    // define all the attributes as numerical
    // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
    // that can be assigned on a per attribute basis

    cv::Mat var_type = cv::Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
    var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

    // this is a classification problem (i.e. predict a discrete number of class
    // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

    var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

    double result; // value returned from a prediction
    rtree = new CvRTrees;

    // load training and testing data sets

    if (loadModel && read_data_from_csv(train_path, training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
        read_data_from_csv(test_path, testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {
      // define the parameters for training the random forest (trees)

      float priors[] = {1,1,1,1};    // weights of each classification for classes
      // (all equal as equal samples of each digit)

      CvRTParams params = CvRTParams(100,   // max depth
                                     1,   // min sample count
                                     0,   // regression accuracy: N/A here
                                     false,   // compute surrogate split, no missing data
                                     30,   // max number of categories (use sub-optimal algorithm for larger numbers)
                                     priors,   // the array of priors
                                     true,    // calculate variable importance
                                     14,         // number of variables randomly selected at node and used to find the best split(s).
                                     100,    // max number of trees in the forest
                                     0.01f,         // forrest accuracy
                                     CV_TERMCRIT_ITER | CV_TERMCRIT_EPS   // termination cirteria
                                     );

      // train random forest classifier (using training data)

      printf( "\nUsing training database: %s\n\n", train_path.c_str());

      rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
                   cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);

      rtree->save(model_path.c_str());
      // perform classifier testing and report results

      cv::Mat test_sample;
      int correct_class = 0;
      int wrong_class = 0;
      int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0};

      printf( "\nUsing testing database: %s\n\n", test_path.c_str());

      for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
      {

        // extract a row from the testing matrix

        test_sample = testing_data.row(tsample);

        // run random forest prediction

        result = rtree->predict(test_sample, cv::Mat());

        printf("Testing Sample %i -> class result (digit %d)\n", tsample, (int) result);

        // if the prediction and the (true) testing classification are the same
        // (N.B. openCV uses a floating point decision tree implementation!)

        if (fabs(result - testing_classifications.at<float>(tsample, 0))
            >= FLT_EPSILON)
        {
          // if they differ more than floating point error => wrong class
          wrong_class++;
          false_positives[(int) result]++;
        }
        else
        {
          // otherwise correct
          correct_class++;
        }
      }

      printf( "\nResults on the testing database: %s\n"
              "\tCorrect classification: %d (%g%%)\n"
              "\tWrong classifications: %d (%g%%)\n",
              argv[2],
              correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
              wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

      for (int i = 0; i < NUMBER_OF_CLASSES; i++)
      {
        printf( "\tClass (digit %d) false postives  %d (%g%%)\n", i,
                false_positives[i],
                (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
      }
    }
    else{
      rtree->load(model_path.c_str());
    }

    cv::resize(frame, small, cv::Size(640, 360), 0, 0, cv::INTER_NEAREST);
    cv_image<bgr_pixel> csmall(small);
    cv::Rect area;
    cv::Size size;

    if (!tracked.empty())
    {
      tracker.update(csmall);
      tracked[0] = grow_rect(tracker.get_position(), 20, 40);
      tracked[0] = tracked[0].intersect(rectangle(640,360));
      area = cv::Rect(2*tracked[0].left(),2*tracked[0].top(),2*tracked[0].width(),2*tracked[0].height());
      track = frame(area).clone();
      size = track.size();
    }
    else
    {
      area = cv::Rect(0,0,1280,720);
      size = cv::Size(640,360);
      track = small.clone();
    }
    cv_image<bgr_pixel> ctrack(track);

    // Detect faces
    std::vector<rectangle> empty;
    std::vector<rectangle> faces = detector(ctrack);
    // Find the pose of each face.
    std::vector<full_object_detection> shapes;
    if (!tracked.empty())
    {
      if (faces.empty())
        tracked.clear();
      else
      {
      }
    }
    else if (!faces.empty())
    {
      tracker.start_track(csmall, faces[0]);
      tracked.push_back(faces[0]);

    }
    for (unsigned long i = 0; i < faces.size(); ++i)
      shapes.push_back(pose_model(ctrack, faces[i]));

    // Display it all on the screen
    render_face_detections(frame, area, size, tracked, faces, shapes, CV_RGB(0,255,0));

    if(fnum++ >= 9)
    {
      t1 = cvGetTickCount();
      fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6);
      t0 = t1; fnum = 0;
    }
    sprintf(text,"%d frames/sec",(int)round(fps));
    cv::putText(frame,text,cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));

    pthread_mutex_lock(frameMut);
    shared = frame.clone();
    pthread_mutex_unlock(frameMut);
  }
  catch(serialization_error& e)
  {
    cout << "You need dlib's default face landmarking model file to run this example." << endl;
    cout << "You can get it from the following URL: " << endl;
    cout << "   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
    cout << endl << e.what() << endl;
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
}
