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
//#include <dlib/gui_widgets.h>
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

using namespace dlib;
using namespace std;
//using namespace cv;



//Constands for finding eye centers

// Debugging
const bool kPlotVectorField = false;

// Algorithm Parameters
const int kFastEyeWidth = 50;
#define PI 3.14159265
// Eye Corner

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
PyObject *pFunc, *pArgs;
uint8_t *pyBufData;
double *pyResData;
cv::RNG rng(12345);
int leftSize = 100, rightSize = 100;
cv::Point RightPupil;
cv::Point leftPupil;
struct sockaddr_un addr;
char buf[1];
int fd,rc;
int counterLeft = 0, counterRight = 0;
char *socket_path = "/var/run/rTreeDemo";
char* train_path = "/Users/Joey/Desktop/RForrest_train/train1.txt";
char* test_path = "/Users/Joey/Desktop/RForrest_train/forrest_test.txt";
int datamode = 0;
int screenPoint = 0;
int train = 0;
double roll, pitch, yaw;
float magicNumber = 1.0865;
CvRTrees* rtree;
float sphereRadius = 13.0382; // Magic number * 12 mm which is the avg radius of eyeball
int useForrest = 0;
int loadModel = 0;
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 8834
#define ATTRIBUTES_PER_SAMPLE 43
#define NUMBER_OF_TESTING_SAMPLES 92
#define NUMBER_OF_CLASSES 4


// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, cv::Mat data, cv::Mat classes,
                       int n_samples )
{
    float tmp;

    // if we can't read the input file then return 0
    FILE* f = fopen( filename, "r" );
    if( !f )
    {
        printf("ERROR: cannot read file %s\n",  filename);
        return 0; // all not OK
    }

    // for each sample in the file

    for(int line = 0; line < n_samples; line++)
    {

        // for each attribute on the line in the file

        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
        {
            if (attribute < 43)
            {

                // first 43 elements (0-43) in each line are the attributes

                fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute) = tmp;
                // printf("%f,", data.at<float>(line, attribute));

            }
            else if (attribute == 43)
            {

                // attribute 44 is the class label {0 ... 9}

                fscanf(f, "%f,", &tmp);
                classes.at<float>(line, 0) = tmp;
                // printf("%f\n", classes.at<float>(line, 0));

            }
        }
    }

    fclose(f);

    return 1; // all OK
}

void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow, double interOcDist) {
  //cvtColor( face, face, CV_BGR2GRAY );
  cv::Mat eyeROIUnscaled = face(eye);
  cv::Mat eyeROI (cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/eye.width) * eye.height),CV_8UC3,pyBufData,cv::Mat::AUTO_STEP);
  scaleToFastSize(eyeROIUnscaled, eyeROI);
  // draw eye region
 // cv::rectangle(face,eye,1234);

  pyResData[0] = (double)eyeROI.rows;
  pyResData[1] = (double)eyeROI.cols;
  pyResData[2] = interOcDist;
  PyObject_CallObject(pFunc, pArgs);
  cv::Point result (pyResData[0],pyResData[1]);
  //imshow(debugWindow,gradientX);
  return unscalePoint(result,eye);
}

// This handles
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    cout<<" In CallBackFunc"<<endl;
     if  ( event == cv::EVENT_LBUTTONDOWN )
     {
          cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
     else if  ( event == cv::EVENT_RBUTTONDOWN )
     {
          cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
     else if  ( event == cv::EVENT_MBUTTONDOWN )
     {
          cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
     else if ( event == cv::EVENT_MOUSEMOVE )
     {
          cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

     }
}

void loadWithPoints(cv::Mat& ip, cv::Mat& img,cv::Point leftPupil,cv::Point RightPupil,std::vector<cv::Point2f> LeftEyePoints, std::vector<cv::Point2f> RightEyePoints, std::vector<cv::Point2f> LeftEllipse, std::vector<cv::Point2f> RightEllipse, cv::Point meanLeft, cv::Point meanRight) {
  int max_d = MAX(img.rows,img.cols);
  float delX = 30.0;
  float delY = 30.0;
  float delZ = 30.0;
  int rightCentre, leftCentre;
  std::vector<cv::Point3f> axisPoints, ModelLeft;

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

  ModelLeft.push_back(cv::Point3f(0,15.5302,17.8295));  // Centre of eye (v 1879)
  ModelLeft.push_back(cv::Point3f(0 +delX,15.5302,17.8295));  // Centre of eye (v 1879)
  ModelLeft.push_back(cv::Point3f(0,15.5302 + delY,17.8295));  // Centre of eye (v 1879)
  ModelLeft.push_back(cv::Point3f(0,15.5302,17.8295 + delZ));  // Centre of eye (v 1879)

  ModelLeftMatrix = cv::Mat(ModelLeft);

  camMatrix = (cv::Mat_<double>(3,3) << max_d, 0, img.cols/2.0,
                    0,  max_d, img.rows/2.0,
                    0,  0,  1.0);

  double _dc[] = {0,0,0,0};
  cv::solvePnP(op,ip,camMatrix,cv::Mat(1,4,CV_64FC1,_dc),rvec,tvec,false,CV_EPNP);

  cv::Mat rotM(3,3,CV_64FC1,rot);
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

   // EYE MODEL
  camMatrix2 = (cv::Mat_<double>(3,3) << max_d, 0, img.cols/2.0,
                    0,  max_d, img.rows/2.0,
                    0,  0,  1.0);

  double _dc2[] = {0,0,0,0};
  cv::solvePnP(op2,ip2,camMatrix2,cv::Mat(1,4,CV_64FC1,_dc),rvec2,tvec2,false,CV_EPNP);

  cv::Mat rotM2(3,3,CV_64FC1,rot2);
  cv::Rodrigues(rvec2,rotM2);
  double* _r2 = rotM2.ptr<double>();
  // printf("rotation mat: \n %.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",
  //   _r[0],_r[1],_r[2],_r[3],_r[4],_r[5],_r[6],_r[7],_r[8]);
  // printf("trans vec: \n %.3f %.3f %.3f\n",tv[0],tv[1],tv[2]);

  double _pm2[12] = {_r2[0],_r2[1],_r2[2],tv2[0],
            _r2[3],_r2[4],_r2[5],tv2[1],
            _r2[6],_r2[7],_r2[8],tv2[2]};

  cv::Matx34d P2(_pm2);
  cv::Mat KP2 = camMatrix2 * cv::Mat(P2);

  cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
  cv::Point2f nose, lEye,rEye,ModelLeftEye;
  cv::RotatedRect lEllipse, rEllipse;

  if(LeftEllipse.size() >= 6){
   lEllipse = fitEllipse(LeftEllipse);
   // ellipse( img, lEllipse, cv::Scalar(255,0,0), 2, 8 );
   }

  if(RightEllipse.size() >= 6){
   rEllipse = fitEllipse(RightEllipse);
   //ellipse( img, rEllipse, cv::Scalar(255,0,0), 2, 8 );
  }

for (int i=0; i<axisMatrix.rows; i++) {
    cv::Mat_<double> X = (cv::Mat_<double>(4,1) << axisMatrix.at<float>(i,0),axisMatrix.at<float>(i,1),axisMatrix.at<float>(i,2),1.0);
    //cout << "object point " << X << endl;
    cv::Mat_<double> opt_p = KP * X;
    cv::Point2f opt_p_img(opt_p(0)/opt_p(2),opt_p(1)/opt_p(2));

    if(i == 0)
      nose = opt_p_img;

    else if(i == 1){
      if(nose == opt_p_img)
        cv::circle(img, opt_p_img, 5, cv::Scalar(255,0,255), 5);

      cv::line(img, nose, opt_p_img, cv::Scalar(255,0,0),5);
    }
    else if(i == 2){
      cv::line(img, nose, opt_p_img, cv::Scalar(0,255,0),5);
    }
    else if(i == 3){
      cv::line(img, nose, opt_p_img, cv::Scalar(0,0,255),5);
    }

    else if(i == 4){
      rEye = opt_p_img;
      //circle(img, opt_p_img, 2, Scalar(255,0,0), 2);
      cv::LineIterator it2(img, opt_p_img, RightPupil, 8);
      rightCentre = opt_p_img.x;
      int counter = 0;
      while(counter < 30){
        it2++;
        counter++;
      }

     // line(img, opt_p_img, it2.pos(), cv::Scalar(0,0,255),5);

    }

    else if(i == 5){

     // line(img, rEye, opt_p_img, cv::Scalar(255,0,0),5);
    }
    else if(i == 6){
     // line(img, rEye, opt_p_img, cv::Scalar(0,255,0),5);
    }
    else if(i == 7){
     // line(img, RightPupil, opt_p_img, cv::Scalar(0,0,255),5);
    }

    else if(i == 8){
      lEye = opt_p_img;
     // circle(img, opt_p_img, 3, Scalar(255,0,0), 2);
      cv::LineIterator it(img, opt_p_img, leftPupil, 8);
      leftCentre = opt_p_img.x;
      int counter = 0;
      while(counter < 30){
        it++;
        counter++;
      }

     // line(img, opt_p_img, it.pos(), cv::Scalar(0,0,255),5);
    }
    else if(i == 9){

     // line(img, lEye, opt_p_img, cv::Scalar(255,0,0),5);
    }
    else if(i == 10){
     // line(img, lEye, opt_p_img, cv::Scalar(0,255,0),5);
    }
    else if(i == 11){
     // line(img, leftPupil, opt_p_img, cv::Scalar(0,0,255),5);
    }
    else if(i == 12){
      cv::LineIterator it(img, opt_p_img, leftPupil, 8);
      leftCentre = opt_p_img.x;
      int counter = 0;
      while(counter < 30){
        it++;
        counter++;
      }
        cv::circle(img, opt_p_img, 2, cv::Scalar(0,255,0), 2);
     //line(img, opt_p_img, it.pos(), cv::Scalar(0,0,255),5);
     // line(img, lEye, opt_p_img, cv::Scalar(0,255,0),5);
    }
    else if(i == 13){
      cv::LineIterator it(img, opt_p_img, RightPupil, 8);
      leftCentre = opt_p_img.x;
      int counter = 0;
      while(counter < 30){
        it++;
        counter++;
      }

     // line(img, opt_p_img, it.pos(), cv::Scalar(0,0,255),5);
      cv::circle(img, opt_p_img, 2, cv::Scalar(0,255,0), 2);
     // line(img, leftPupil, opt_p_img, cv::Scalar(0,0,255),5);
    }
  }


for (int i=0; i<ModelLeftMatrix.rows; i++) {
    cv::Mat_<double> X2 = (cv::Mat_<double>(4,1) << ModelLeftMatrix.at<float>(i,0),ModelLeftMatrix.at<float>(i,1),ModelLeftMatrix.at<float>(i,2),1.0);
    //cout << "object point " << X << endl;
    cv::Mat_<double> opt_p2 = KP2 * X2;
    cv::Point2f opt_p_img2(opt_p2(0)/opt_p2(2),opt_p2(1)/opt_p2(2));

    if(i == 0)
      ModelLeftEye = opt_p_img2;

    else if(i == 1){
     // cv::line(img, ModelLeftEye, opt_p_img2, cv::Scalar(255,0,0),5);
    }
    else if(i == 2){
     // cv::line(img, ModelLeftEye, opt_p_img2, cv::Scalar(0,255,0),5);
    }
    else if(i == 3){
     // cv::line(img, ModelLeftEye, opt_p_img2, cv::Scalar(0,0,255),5);
    }
  }

  interOcDist = abs(leftCentre - rightCentre);
    // cout<<"Inter occular Distance: "<<interOcDist<<endl;
  cv::Point2f leftLeftEdge, leftRightEdge, rightLeftEdge, rightRightEdge; // convention is lower case denotes which eye , upper case denotes left or right edge
  //reproject object points - check validity of found projection matrix
  for (int i=0; i<op.rows; i++) {
    cv::Mat_<double> X = (cv::Mat_<double>(4,1) << op.at<float>(i,0),op.at<float>(i,1),op.at<float>(i,2),1.0);
    cv::Mat_<double> opt_p = KP * X;
    cv::Point2f opt_p_img(opt_p(0)/opt_p(2),opt_p(1)/opt_p(2));

    if(i == 1){
      leftLeftEdge.x = opt_p_img.x;
      leftLeftEdge.y = opt_p_img.y;
    }
    else if( i == 2){
      leftRightEdge.x = opt_p_img.x;
      leftRightEdge.y = opt_p_img.y;
    }
    else if( i == 3){
      rightLeftEdge.x = opt_p_img.x;
      rightLeftEdge.y = opt_p_img.y;
    }
    else if( i == 4){
      rightRightEdge.x = opt_p_img.x;
      rightRightEdge.y = opt_p_img.y;
    }

   // circle(img, opt_p_img, 5, Scalar(0,0,255), 5);
  }

  double x, ret, val;
  // cv::KalmanFilter KF(4, 2, 0);
  // KF.statePre.at<float>(0) = roll;
  // KF.statePre.at<float>(1) = pitch;
  // KF.statePre.at<float>(2) = yaw;
  // KF.statePre.at<float>(3) = 0;
  // KF.transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
  // cv::Mat_<float> measurement(3,1); measurement.setTo(cv::Scalar(0));
  // setIdentity(KF.measurementMatrix);
  // setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
  // setIdentity(KF.measurementNoiseCov, cv::Scalar::all(10));
  // setIdentity(KF.errorCovPost, cv::Scalar::all(.1));
  val = 180.0 / PI;

  yaw = atan2 (_r[3], _r[0]) * val;
  pitch = atan2(-1*_r[6],sqrt(_r[7]*_r[7] + _r[8]*_r[8])) * val;
  roll = atan(_r[7]/_r[8]) * val;
  // cv::Mat prediction = KF.predict();
  // measurement(0) = roll;
  // measurement(1) = pitch;
  // measurement(2) = yaw;
  // cv::Mat estimated = KF.correct(measurement);
  // roll = estimated.at<float>(0);
  // pitch = estimated.at<float>(1);
  // yaw = estimated.at<float>(2);
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

  // cv::circle(img, LeftEyePoints[0], 3, cv::Scalar(255,0,0), 3);
  // cv::circle(img, LeftEyePoints[2], 3, cv::Scalar(255,0,0), 3);
  // cout << " Eye Corner: "<<LeftEyeCornerAvg <<endl;
  // cout << " Pupil Height:"<<leftPupil.y <<endl;
  // cout<<" ROLL: "<<roll<<endl;
  // cout<<" PITCH: "<<pitch<<endl;
   //cout<<" YAW: "<<yaw<<endl;

  leftSize = int(leftRightEdge.x - leftLeftEdge.x);
  rightSize = int(rightRightEdge.x - rightLeftEdge.x);
  int xDiff = leftPupil.x - lEllipse.center.x;
 //s cout<<xDiff<<endl;
  int prob;

  // circle(img, Point (middle.x,LeftEyeCornerAvg), 5, Scalar(255,0,0), 5);
  // circle(img, Point (middle.x,leftPupil.y), 5, Scalar(255,255,0), 5);

  //  if(pitch > 0){
  //   if( roll > 0){
  //     // prob = (std::rand() % 100 + 1);
  //     // cout<<prob<<endl;
  //     if(leftPupil.x <lEllipse.center.x){
  //        cv::circle(img, BotLeft, 10, cv::Scalar(0,0,255), 10);
  //        strcpy (buf,"3");
  //     }
  //     else{
  //       counterRight++;
  //      // if(counterRight >= 3){
  //       cv::circle(img, BotRight, 10, cv::Scalar(0,0,255), 10);
  //       strcpy (buf,"4");
  //       //counterLeft = 0;

  //     }
  //     //cv::circle(img, BotLeft, 10, cv::Scalar(0,0,255), 10);
  //   }
  //   else{
  //     if(leftPupil.x < lEllipse.center.x){
  //       cv::circle(img, TopLeft, 10, cv::Scalar(0,0,255), 10);
  //       strcpy (buf,"1");
  //     }
  //     else{
  //       cv::circle(img, TopRight, 10, cv::Scalar(0,0,255), 10);
  //       strcpy (buf,"2");
  //     }
  //  //  cv::circle(img, TopLeft, 10, cv::Scalar(255,0,0), 10);
  //     }
  //  }
  //  else{
  //   if(roll > 0){
  //     if(RightPupil.x < rEllipse.center.x){
  //        cv::circle(img, BotLeft, 10, cv::Scalar(0,0,255), 10);
  //        strcpy (buf,"3");
  //     }
  //     else{
  //        cv::circle(img, BotRight, 10, cv::Scalar(0,255,0), 10);
  //        strcpy (buf,"4");
  //     }
  //   }
  //   else{
  //     if(RightPupil.x < rEllipse.center.x){
  //        cv::circle(img, TopLeft, 10, cv::Scalar(0,0,255), 10);
  //        strcpy (buf,"1");
  //     }
  //     else{
  //       cv::circle(img, TopRight, 10, cv::Scalar(0,255,255), 10);
  //       strcpy (buf,"2");
  //     }
  //   }
  // }



  //  if(pitch > 0){
  //   if( roll > 0){
  //     cv::circle(img, BotLeft, 10, cv::Scalar(0,0,255), 10);
  //   }
  //   else{
  //     cv::circle(img, TopLeft, 10, cv::Scalar(0,0,255), 10);
  //     }
  //  }
  //  else{
  //   if(roll > 0){
  //     cv::circle(img, BotRight, 10, cv::Scalar(0,0,255), 10);
  //   }
  //   else{
  //     cv::circle(img, TopRight, 10, cv::Scalar(0,0,255), 10);
  //   }
  // }

 // write(fd, buf, sizeof(buf));
 // cv::circle(img, lEllipse.center, 2, cv::Scalar(255,0,255), 2);
 // cv::circle(img, rEllipse.center, 2, cv::Scalar(255,0,255), 2);

  float yDiff = abs(leftPupil.y - lEllipse.center.y);
  //cout<<"Difference is"<<yDiff<<endl;

  // if(leftPupil.x <= lEllipse.center.x)
  //     cv::circle(img, TopLeft, 10, cv::Scalar(0,0,255), 10);
  // else
  //     cv::circle(img, TopRight, 10, cv::Scalar(0,0,255), 10);

  // if(yDiff > 4){
  //   if(leftPupil.x < lEllipse.center.x)
  //     cv::circle(img, TopLeft, 10, cv::Scalar(0,0,255), 10);
  //   else
  //     cv::circle(img, TopRight, 10, cv::Scalar(0,0,255), 10);
  // }
  // else{
  //   if(leftPupil.x < lEllipse.center.x)
  //     cv::circle(img, BotLeft, 10, cv::Scalar(0,0,255), 10);
  //   else
  //     cv::circle(img, BotRight, 10, cv::Scalar(0,0,255), 10);
  // }

  float mapX, mapY;

  // mapX = img.cols/ lEllipse.size.width;
  // mapY = img.rows/ lEllipse.size.height;

  cv::Point2f gazePoint;


  // if(pitch > 0){
  //   if(lEllipse.size.width >= 21)
  //     cv::circle(img, TopLeft, 10, cv::Scalar(255,0,0), 10);
  //   else
  //     cv::circle(img, BotLeft, 10, cv::Scalar(0,0,255), 10);
  // }
  // else{
  //   if(lEllipse.size.width >= 21)
  //     cv::circle(img, TopRight, 10, cv::Scalar(0,255,255), 10);
  //   else
  //     cv::circle(img, BotRight, 10, cv::Scalar(0,255,0), 10);
  // }

  //strcpy (buf,"3");
 // write(fd, buf, sizeof(buf));

  // cout << " Left Size"<<leftSize<<endl;
  // cout << " Right Size"<<rightSize<<endl;
  rotM = rotM.t();// transpose to conform with majorness of opengl matrix

}

 std::vector<double> gazeStabilizer;

void render_face_detections (cv::Mat& image, cv::Rect& area, cv::Size& size, const std::vector<rectangle>& trackers, const std::vector<rectangle>& faces, const std::vector<full_object_detection>& dets, const cv::Scalar color)
{
  // Variables
  cv::Mat newImg, src_gray,crop,crop2, eyeMap ,leftCrop, rightCrop;
  newImg= image.clone();
  //eyeMap = image.clone();
  std::vector<cv::Point2f> srcImagePoints, LeftEyePoints, RightEyePoints, LeftEyePointsModel,LeftEllipse, RightEllipse;
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

  cv::setMouseCallback("Face tracker", CallBackFunc, NULL);
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

        for(unsigned long i =0 ; i<=67; i++)
        {
            p1.x = d.part(i).x()*xScale+area.x;
            p1.y = d.part(i).y()*yScale+area.y;
            std::string x = to_string(i);

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

           // cv::putText(image,x,p1, CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
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
                   // LeftEyePoints.push_back(leftPupil);
                    // RightCorner.x = leftPupil.x - 15*leftSize/100;
                    // RightCorner.y = leftPupil.y;
                    // LeftCorner.x = leftPupil.x + 15*leftSize/100;
                    // LeftCorner.y = leftPupil.y;
                    // TopCorner.x = leftPupil.x;
                    // TopCorner.y = leftPupil.y - 15*leftSize/100;
                    // BotCorner.x = leftPupil.x;
                    // BotCorner.y = leftPupil.y + 15*leftSize/100;
                    // LeftEyePointsModel.push_back(leftPupil);
                    // //cout<<RightCorner<<endl;
                    // LeftEyePointsModel.push_back(RightCorner);
                    // LeftEyePointsModel.push_back(LeftCorner);
                    // LeftEyePointsModel.push_back(TopCorner);
                    // LeftEyePointsModel.push_back(BotCorner);
                   // circle(image,leftPupil,2, cv::Scalar(0,255,0), 2);

                }

                 if(i == 39)
                {
                    LeftEyePoints.push_back(p1);
                   // cv::line(image, p1, leftPupil, Scalar(0,0,255),5);
                }


                if(i == 42){
                    RightEyePoints.push_back(p1);
                    RightEye.x = p1.x - 5*rightSize/100;
                    RightEye.y = p1.y - 50*rightSize/100;
                    RightEye.width = rightSize;
                    RightEye.height = rightSize;

                  try{
                      crop2 = newImg(RightEye);
                      RightPupil = findEyeCenter(newImg,RightEye,"Right Eye",interOcDist);
                  }
                    catch(...){
                      cout<<"RIGHT EYE EXCEPTION HERE !!!!!!!!!!!!!!!!!!!!"<<endl;
                    }
                    RightPupil.x += RightEye.x;
                    RightPupil.y += RightEye.y;
                    RightEyePoints.push_back(RightPupil);
                 // //   cv::line(image, p1, RightPupil, Scalar(0,0,255),5);
                  //   cv::circle(image,RightPupil,2, cv::Scalar(0,255,0), 2);
                  }

                if (i == 45){
                    RightEyePoints.push_back(p1);
                   // cv::line(image, p1, RightPupil, Scalar(0,0,255),5);
                  }

            }
        }
    }

  cv::RotatedRect lEllipse, rEllipse;
  float startX = 10000, startY = 10000;
  float right_startX = 10000, right_startY = 10000;

  if(LeftEllipse.size() >= 6){
   for(unsigned long i = 0; i < LeftEyePoints.size(); i++)
   {
      if(LeftEyePoints[i].x < startX)
        startX = LeftEyePoints[i].x;
      if(LeftEyePoints[i].y < startY)
        startY = LeftEyePoints[i].y;
   }

   rectLeft.x = startX - 10*leftSize/100;
   rectLeft.y = startY - 20*leftSize/100;
   lEllipse = fitEllipse(LeftEllipse);
   rectLeft.width = lEllipse.size.height + 10*leftSize/100;
   rectLeft.height = 3*lEllipse.size.height/4;
   //cout<<"X coord "<<rectLeft.x<<"Y coord "<<rectLeft.y<<endl;
   //leftCrop = newImg(rectLeft);
    try
      {
        leftCrop = newImg(rectLeft);
     //   cv::imshow("Left Crop",leftCrop);
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
       // LeftEyePointsModel.push_back(lEllipse.center);
        circle(image,leftPupil,2, cv::Scalar(255,125,255), 2);
     //   circle(image,RightCorner,2, cv::Scalar(0,255,255), 2);
     //   circle(image,LeftCorner,2, cv::Scalar(0,255,255), 2);
     //   circle(image,BotCorner,2, cv::Scalar(0,255,255), 2);
     //   circle(image,TopCorner,2, cv::Scalar(0,255,255), 2);

      }
  catch(...)
      {
        cout<<"LEFT BOUNDING BOX EXCEPTION"<<endl;
      }
   }

  if(RightEllipse.size() >= 6){
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
 //  cout<<"X coord "<<rectLeft.x<<"Y coord "<<rectLeft.y<<endl;

   try
      {
        rightCrop = newImg(rectRight);
        //cv::imshow("Right Crop",rightCrop);
        //cv::imwrite("/Users/Joey/Desktop/failure_case.jpg", rightCrop);
        RightPupil = findEyeCenter(newImg,rectRight,"Left Eye",interOcDist);
        RightPupil.x += rectRight.x;
        RightPupil.y += rectRight.y;
        circle(image,RightPupil,2, cv::Scalar(0,0,255), 2);

      }
  catch(...)
      {
        cout<<"Right BOUNDING BOX EXCEPTION"<<endl;
      }
   //cv::imshow("Right Crop", rightCrop);

  }
    ip = cv::Mat(srcImagePoints);
    ip2 = cv::Mat(LeftEyePointsModel);

  if(LeftEllipse.size() == 6){
    cv::Mat mean_;
    cv::reduce(LeftEllipse, mean_, CV_REDUCE_AVG, 1);
    // convert from Mat to Point - there may be even a simpler conversion,
    // but I do not know about it.
    cv::Point2f mean(mean_.at<float>(0,0), mean_.at<float>(0,1));
    meanLeft.x = mean.x;
    meanLeft.y = mean.y;

    //circle(image,mean,2, cv::Scalar(0,255,255), 2);
  }

  if(RightEllipse.size() == 6){
    cv::Mat mean_;
    cv::reduce(RightEllipse, mean_, CV_REDUCE_AVG, 1);
    cv::Point2f mean(mean_.at<float>(0,0), mean_.at<float>(0,1));
    meanRight.x = mean.x;
    meanRight.y = mean.y;

    //circle(image,mean,2, cv::Scalar(255,255,255), 2);
  }

  cv::Point leftCentre, rightCentre, leftDisplacement, rightDisplacement;
  leftCentre.x = (LeftEyePoints[0].x + LeftEyePoints[2].x)/2;
  leftCentre.y = (LeftEyePoints[0].y + LeftEyePoints[2].y)/2;
  leftDisplacement.x = leftPupil.x - lEllipse.center.x;
  leftDisplacement.y = leftPupil.y - lEllipse.center.y;
  rightDisplacement.x = RightPupil.x - rEllipse.center.x;
  rightDisplacement.y = RightPupil.y - rEllipse.center.y;
 // ofstream outputFile ("/Users/Joey/Desktop/output.txt");
  std::ofstream leftTopFile,leftBotFile, rightTopFile, rightBotFile, trainForrest;
  cv::Mat var_matrix;
  cv::Mat gazeMatrix= cv::Mat(1, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  double result;
  if(LeftEllipse.size() == 6 && RightEllipse.size() == 6 && useForrest == 1){
      int attribute_counter = 0;
     // auto first = true;
      for (std::vector<cv::Point2f>::iterator i = LeftEllipse.begin(); i != LeftEllipse.end(); ++i)
      {

          gazeMatrix.at<float>(0, attribute_counter) = i->x;
          attribute_counter++;
          gazeMatrix.at<float>(0, attribute_counter) = i->y;
          attribute_counter++;
      }

      //trainForrest <<",";
      //auto second = true;
      for (std::vector<cv::Point2f>::iterator i = RightEllipse.begin(); i != RightEllipse.end(); ++i)
      {
          gazeMatrix.at<float>(0, attribute_counter) = i->x;
          attribute_counter++;
          gazeMatrix.at<float>(0, attribute_counter) = i->y;
          attribute_counter++;

      }

    gazeMatrix.at<float>(0, attribute_counter) = roll;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = pitch;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = yaw;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = meanLeft.x;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = meanLeft.y;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = leftDisplacement.x;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = leftDisplacement.y;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = rightDisplacement.x;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = rightDisplacement.y;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = leftPupil.x;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = leftPupil.y;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = lEllipse.center.x;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = lEllipse.center.y;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = meanRight.x;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = meanRight.y;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = RightPupil.x;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = RightPupil.y;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = rEllipse.center.x;
    attribute_counter++;
    gazeMatrix.at<float>(0, attribute_counter) = rEllipse.center.y;
    attribute_counter++;
    result = rtree->predict(gazeMatrix, cv::Mat());
    gazeStabilizer.push_back(result);
  }

  int mode;
  if(gazeStabilizer.size() > 7){
    gazeStabilizer.erase(gazeStabilizer.begin());
    std::vector<int> histogram(8,0);
    for( int i=0; i<7; ++i )
      ++histogram[ gazeStabilizer[i] ];
    mode = std::max_element( histogram.begin(), histogram.end() ) - histogram.begin();
  }

  //cout<<"Mode is "<<mode<<" ";
  // for(int i=0; i<gazeStabilizer.size(); ++i)
  //   std::cout << gazeStabilizer[i] << ' ';
  // cout<<endl;

  if(result != mode)
    result = mode;

  if(result == 0 && useForrest == 1){
    cv::circle(image, TopLeft, 10, cv::Scalar(0,255,0), 10);
    strcpy (buf,"1");
    write(fd, buf, sizeof(buf));
  }
  else if(result == 1 && useForrest == 1){
    cv::circle(image, TopRight, 10, cv::Scalar(0,255,0), 10);
    strcpy (buf,"2");
    write(fd, buf, sizeof(buf));
  }
  else if(result == 2 && useForrest == 1){
    cv::circle(image, BotLeft, 10, cv::Scalar(0,255,0), 10);
    strcpy (buf,"3");
    write(fd, buf, sizeof(buf));
  }
  else if(result == 3 && useForrest == 1){
    cv::circle(image, BotRight, 10, cv::Scalar(0,255,0), 10);
    strcpy (buf,"4");
    write(fd, buf, sizeof(buf));
  }

// Training Data Collection interface for Random Forrests
  if(LeftEllipse.size() == 6 && RightEllipse.size() == 6 && screenPoint == 1){
    circle(image,cv::Point(10 ,10), 5, cv::Scalar(0,255,0), 5);
    if(train == 0)
      trainForrest.open("/Users/Joey/Desktop/RForrest_train/new_train2.txt", std::ios_base::app);
    else
      trainForrest.open("/Users/Joey/Desktop/RForrest_train/forrest_test.txt", std::ios_base::app);

      auto first = true;
      for (std::vector<cv::Point2f>::iterator i = LeftEllipse.begin(); i != LeftEllipse.end(); ++i)
      {
          if (!first) { trainForrest << ",";}
          first = false;
          trainForrest << i->x <<","<<i->y;
      }

      //trainForrest <<",";
      auto second = true;
      for (std::vector<cv::Point2f>::iterator i = RightEllipse.begin(); i != RightEllipse.end(); ++i)
      {
          if (!first) { trainForrest << ",";}
          second = false;
          trainForrest << i->x <<","<<i->y;
      }

    trainForrest<<","<<roll<<","<<pitch<<","<<yaw<<",";
    trainForrest<< meanLeft.x <<"," << meanLeft.y<<",";
    trainForrest<<leftDisplacement.x<<","<<leftDisplacement.y<<",";
    trainForrest<<rightDisplacement.x<<","<<rightDisplacement.y<<",";
    trainForrest<<leftPupil.x<<","<<leftPupil.y<<",";
    trainForrest<<lEllipse.center.x<<","<<lEllipse.center.y<<",";
    trainForrest<< meanRight.x <<"," << meanRight.y<<",";
    trainForrest<<RightPupil.x<<","<<RightPupil.y<<",";
    trainForrest<<rEllipse.center.x<<","<<rEllipse.center.y<<",";
    trainForrest<<screenPoint -1;
    trainForrest << endl;
  }

  if(LeftEllipse.size() == 6 && RightEllipse.size() == 6 && screenPoint == 2){
    circle(image,cv::Point(image.cols -10 ,10), 5, cv::Scalar(0,255,0), 5);
    if(train == 0)
      trainForrest.open("/Users/Joey/Desktop/RForrest_train/new_train2.txt", std::ios_base::app);
    else
      trainForrest.open("/Users/Joey/Desktop/RForrest_train/forrest_test.txt", std::ios_base::app);

      auto first = true;
      for (std::vector<cv::Point2f>::iterator i = LeftEllipse.begin(); i != LeftEllipse.end(); ++i)
      {
          if (!first) { trainForrest << ",";}
          first = false;
          trainForrest << i->x <<","<<i->y;
      }

      //trainForrest <<",";
      auto second = true;
      for (std::vector<cv::Point2f>::iterator i = RightEllipse.begin(); i != RightEllipse.end(); ++i)
      {
          if (!first) { trainForrest << ",";}
          second = false;
          trainForrest << i->x <<","<<i->y;
      }

    trainForrest<<","<<roll<<","<<pitch<<","<<yaw<<",";
    trainForrest<< meanLeft.x <<"," << meanLeft.y<<",";
    trainForrest<<leftDisplacement.x<<","<<leftDisplacement.y<<",";
    trainForrest<<rightDisplacement.x<<","<<rightDisplacement.y<<",";
    trainForrest<<leftPupil.x<<","<<leftPupil.y<<",";
    trainForrest<<lEllipse.center.x<<","<<lEllipse.center.y<<",";
    trainForrest<< meanRight.x <<"," << meanRight.y<<",";
    trainForrest<<RightPupil.x<<","<<RightPupil.y<<",";
    trainForrest<<rEllipse.center.x<<","<<rEllipse.center.y<<",";
    trainForrest<<screenPoint -1;
    trainForrest << endl;
  }


  if(LeftEllipse.size() == 6 && RightEllipse.size() == 6 && screenPoint == 3){
    circle(image,cv::Point(10,image.rows - 10), 5, cv::Scalar(0,255,0), 5);
    if(train == 0)
      trainForrest.open("/Users/Joey/Desktop/RForrest_train/new_train2.txt", std::ios_base::app);
    else
      trainForrest.open("/Users/Joey/Desktop/RForrest_train/forrest_test.txt", std::ios_base::app);

    auto first = true;
      for (std::vector<cv::Point2f>::iterator i = LeftEllipse.begin(); i != LeftEllipse.end(); ++i)
      {
          if (!first) { trainForrest << ",";}
          first = false;
          trainForrest << i->x <<","<<i->y;
      }

      //trainForrest <<",";
      auto second = true;
      for (std::vector<cv::Point2f>::iterator i = RightEllipse.begin(); i != RightEllipse.end(); ++i)
      {
          if (!first) { trainForrest << ",";}
          second = false;
          trainForrest << i->x <<","<<i->y;
      }

    trainForrest<<","<<roll<<","<<pitch<<","<<yaw<<",";
    trainForrest<< meanLeft.x <<"," << meanLeft.y<<",";
    trainForrest<<leftDisplacement.x<<","<<leftDisplacement.y<<",";
    trainForrest<<rightDisplacement.x<<","<<rightDisplacement.y<<",";
    trainForrest<<leftPupil.x<<","<<leftPupil.y<<",";
    trainForrest<<lEllipse.center.x<<","<<lEllipse.center.y<<",";
    trainForrest<< meanRight.x <<"," << meanRight.y<<",";
    trainForrest<<RightPupil.x<<","<<RightPupil.y<<",";
    trainForrest<<rEllipse.center.x<<","<<rEllipse.center.y<<",";
    trainForrest<<screenPoint -1;
    trainForrest << endl;
  }

  if(LeftEllipse.size() == 6 && RightEllipse.size() == 6 && screenPoint == 4){
    circle(image,cv::Point(image.cols -10 ,image.rows - 10), 5, cv::Scalar(0,255,0), 5);
    if(train == 0)
      trainForrest.open("/Users/Joey/Desktop/RForrest_train/new_train2.txt", std::ios_base::app);
    else
      trainForrest.open("/Users/Joey/Desktop/RForrest_train/forrest_test.txt", std::ios_base::app);

      auto first = true;
      for (std::vector<cv::Point2f>::iterator i = LeftEllipse.begin(); i != LeftEllipse.end(); ++i)
      {
          if (!first) { trainForrest << ",";}
          first = false;
          trainForrest << i->x <<","<<i->y;
      }

      //trainForrest <<",";
      auto second = true;
      for (std::vector<cv::Point2f>::iterator i = RightEllipse.begin(); i != RightEllipse.end(); ++i)
      {
          if (!first) { trainForrest << ",";}
          second = false;
          trainForrest << i->x <<","<<i->y;
      }

    trainForrest<<","<<roll<<","<<pitch<<","<<yaw<<",";
    trainForrest<< meanLeft.x <<"," << meanLeft.y<<",";
    trainForrest<<leftDisplacement.x<<","<<leftDisplacement.y<<",";
    trainForrest<<rightDisplacement.x<<","<<rightDisplacement.y<<",";
    trainForrest<<leftPupil.x<<","<<leftPupil.y<<",";
    trainForrest<<lEllipse.center.x<<","<<lEllipse.center.y<<",";
    trainForrest<< meanRight.x <<"," << meanRight.y<<",";
    trainForrest<<RightPupil.x<<","<<RightPupil.y<<",";
    trainForrest<<rEllipse.center.x<<","<<rEllipse.center.y<<",";
    trainForrest<<screenPoint -1;
    trainForrest << endl;
  }

  if(srcImagePoints.size() == 7 && LeftEyePointsModel.size() == 5){
         loadWithPoints(ip,image,leftPupil,RightPupil,LeftEyePoints,RightEyePoints, LeftEllipse, RightEllipse, meanLeft, meanRight);
     }

  //line(image, LeftEyePoints[0], LeftEyePoints[2], Scalar(0,255,0),5);
  rightCentre.x = (RightEyePoints[0].x + RightEyePoints[2].x)/2;
  rightCentre.y = (RightEyePoints[0].y + RightEyePoints[2].y)/2;

 // cv::line(image, leftCentre, leftPupil, Scalar(255,0,0),5);
  srcImagePoints.clear();
  LeftEyePointsModel.clear();
  LeftEllipse.clear();
  LeftEyePoints.clear();
  RightEyePoints.clear();

}


pthread_mutex_t _frameMut, *frameMut=&_frameMut;
cv::Mat *sharePtr = NULL;
int userInput = 0;

void glfwErrorCB(int error, const char* description)
{
    fputs(description, stderr);
}

void glfwKeyCB(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, GL_TRUE);
            userInput = 1;
        }
        // else if(key == GLFW_KEY_W)
        // {
        //   datamode = 1;
        //   cout<< "Changed Datamode to "<<datamode<<endl;
        // }
        //  else if(key == GLFW_KEY_S)
        // {
        //   datamode = 2;
        //   cout<< "Changed Datamode to "<<datamode<<endl;
        // }
         else if(key == GLFW_KEY_R)
        {
          datamode = 0;
          screenPoint = 0;
          cout<< "Changed Datamode to "<<datamode<<endl;
        }

        else if(key == GLFW_KEY_U)
        {
          useForrest = 1;
        }
        else if(key == GLFW_KEY_I)
        {
          useForrest = 0;
        }

        else if(key == GLFW_KEY_Q)
        {
          screenPoint = 1;
        }

        else if(key == GLFW_KEY_W)
        {
          screenPoint = 2;
        }
        else if(key == GLFW_KEY_A)
        {
          screenPoint = 3;
        }
        else if(key == GLFW_KEY_S)
        {
          screenPoint = 4;
        }
        else if(key == GLFW_KEY_T)
        {
          train = 1;
        }
        else if(key == GLFW_KEY_Y)
        {
          train = 0;
        }
    }
}

void *GLloop(void* arg)
{
    GLFWwindow* window = (GLFWwindow *) arg;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    GLuint texture;
    glGenTextures (1, &texture);
    glBindTexture (GL_TEXTURE_2D, texture);
    glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    while (!glfwWindowShouldClose(window))
    {
        float ratio;
        int width, height;

        glfwGetFramebufferSize(window, &width, &height);

        pthread_mutex_lock(frameMut);
        if (sharePtr != NULL && sharePtr->rows > 0)
        {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sharePtr->cols, sharePtr->rows, 0,
                         GL_BGR, GL_UNSIGNED_BYTE, sharePtr->data);
            pthread_mutex_unlock(frameMut);

            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

            ratio = ((float)sharePtr->cols/width) / ((float)sharePtr->rows/height);
            if (ratio > 1)
                glOrtho(-1.f, 1.f, -ratio, ratio, 1.f, -1.f);
            else
                glOrtho(-1/ratio, 1/ratio, -1.f, 1.f, 1.f, -1.f);

            glLoadIdentity();
            glEnable(GL_TEXTURE_2D);
            glBegin(GL_QUADS);
            glTexCoord2i(0, 0); glVertex2i(-1,  1);
            glTexCoord2i(1, 0); glVertex2i( 1,  1);
            glTexCoord2i(1, 1); glVertex2i( 1, -1);
            glTexCoord2i(0, 1); glVertex2i(-1, -1);
            glEnd();

            glfwSwapBuffers(window);
        }
        else
            pthread_mutex_unlock(frameMut);

    }
    pthread_exit(NULL);
}


int main(int argc, const char **argv)
{
    /*****************************************************************************/

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

  PyObject *pName, *pModule, *pIn, *pOut;

  Py_Initialize();
  PyRun_SimpleString("import sys");
    strcpy(syspath, "sys.path.append('");
    if (getcwd(pwd, sizeof(pwd)) != NULL)
        strcat(syspath, pwd);
    strcat(syspath, "')");
    PyRun_SimpleString(syspath);
  cout<<syspath<<endl;
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

  /********************** Unix Socket Initialization START ****************************/

  if ( (fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
    perror("socket error");
    exit(-1);
  }

  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path)-1);

  if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    perror("connect error");
    exit(-1);
  }
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

  tv[0]=0;tv[1]=0;tv[2]=1;
  tvec = cv::Mat(tv);

  tv2[0]=0;tv2[1]=0;tv2[2]=1;
  tvec2 = cv::Mat(tv2);

  camMatrix = cv::Mat(3,3,CV_64FC1);
  camMatrix2 = cv::Mat(3,3,CV_64FC1);

    try
    {
        cv::VideoCapture cap(0);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

        std::vector<cv::Mat> frames;
       // cv::VideoWriter video("/Users/Joey/Desktop/Joey.avi",CV_FOURCC('8','B','P','S'),10, cv::Size(1280,720),true);
        //outputVideo.open(PathName, CV_FOURCC('8','B','P','S'), myvideo.get(CV_CAP_PROP_FPS),CvSize(outputframe.size()));

        glfwSetErrorCallback(glfwErrorCB);
        if (!glfwInit())
            exit(EXIT_FAILURE);

        window = glfwCreateWindow(1920, 1080, "Face tracker", NULL, NULL);
        if (!window)
        {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwSetKeyCallback(window, glfwKeyCB);

        pthread_mutex_init(frameMut, NULL);
        pthread_attr_init(tAttr);
        pthread_attr_setdetachstate (tAttr, PTHREAD_CREATE_JOINABLE);
        pthread_create(glThread, tAttr, GLloop, window);

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        correlation_tracker tracker;

        // Grab and process frames until the main window is closed by the user.
        int64 t1, t0 = cvGetTickCount();
        int fnum=0;
        double fps=0;
        cv::Mat shared;
        sharePtr = &shared;

        cv::Mat frame, small, track;
        cap >> frame;
        cv::flip(frame, frame, 1);
        cv_image<bgr_pixel> cframe(frame);

        std::vector<rectangle> tracked;

        int fCount = 0;

        if (argc == 2)
        {
            printf ("Running frame rate test for 100 frames\n");
            printf ("(playing back results while collecting frames)\n");
            frames[fCount++] = frame.clone();
        }

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

    if (loadModel == 0 && read_data_from_csv(train_path, training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(test_path, testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {
        // define the parameters for training the random forest (trees)

        float priors[] = {1,1,1,1};  // weights of each classification for classes
        // (all equal as equal samples of each digit)

        CvRTParams params = CvRTParams(100, // max depth
                                       14, // min sample count
                                       0, // regression accuracy: N/A here
                                       false, // compute surrogate split, no missing data
                                       30, // max number of categories (use sub-optimal algorithm for larger numbers)
                                       priors, // the array of priors
                                       true,  // calculate variable importance
                                       40,       // number of variables randomly selected at node and used to find the best split(s).
                                       1000,  // max number of trees in the forest
                                       0.01f,       // forrest accuracy
                                       CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria
                                      );

        // train random forest classifier (using training data)

        printf( "\nUsing training database: %s\n\n", train_path);

        rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
                     cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);

        rtree->save("Random_forrest_classifier.yml");


        // perform classifier testing and report results

        cv::Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0};

        printf( "\nUsing testing database: %s\n\n", test_path);

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
      rtree->load("Random_forrest_classifier.yml");
    }
        while(1)
        {
            cv::resize(frame, small, cv::Size(640, 360), 0, 0, cv::INTER_NEAREST);
            cv_image<bgr_pixel> csmall(small);
            cv::Rect area;
            cv::Size size;
              //set the callback function for any mouse event
           // cv::setMouseCallback("Face tracker", CallBackFunc, NULL);

            if (!tracked.empty())
            {
                tracker.update(csmall);
                tracked[0] = grow_rect(tracker.get_position(), 20, 40);
                tracked[0] = tracked[0].intersect(rectangle(640,360));
                area = cv::Rect(2*tracked[0].left(),2*tracked[0].top(),2*tracked[0].width(),2*tracked[0].height());
                track = frame(area).clone();
                size = track.size();
                //size = cv::Size(240,300);
               // cv::resize(frame(area).clone(), track, size, 0, 0, cv::INTER_LINEAR);
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

            glfwPollEvents();
            cap >> frame;
            cv::flip(frame, frame, 1);
            //frame = frame + cv::Scalar(50, 50, 50); //decrease the brightness by 75 units
        //    video.write(frame);
            if (userInput == 1)
                break;
        }
        pthread_mutex_unlock(frameMut);
        pthread_join(*glThread,NULL);
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

