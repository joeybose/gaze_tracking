#include "Features.h"

const std::string PoseModelPath = "models/shape_predictor_68_face_landmarks.dat";
const std::string GazeTrackerModelPath = "models/normalized_Random_forrest_classifier.yml";
const int LeftSize = 100;
const int RightSize = 100;
const int kFastEyeWidth = 50;

// global definitions (for speed and ease of use)
#define ATTRIBUTES_PER_SAMPLE 47

GazeMatrix::GazeMatrix() {
  _setupModel();
  _setupIrisDetector();
}

void GazeMatrix::_setupModel() {
  std::vector<cv::Point3f> modelPoints, eyeModel;

  eyeModel.push_back(cv::Point3f(0,15.5302,17.8295));  // Centre of eye (v 1879)
  eyeModel.push_back(cv::Point3f(6.52994,15.5302,15.3021));  //Right Eye corner
  eyeModel.push_back(cv::Point3f(-6.52994, 15.5302, 15.302));  // Left Corner (v 1879)
  eyeModel.push_back(cv::Point3f(0, 22.0601, 15.302));  // Top Corner (v 1879)
  eyeModel.push_back(cv::Point3f(0, 9.0023, 15.302));  // Bot Corner (v 1879)
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

  mOp = cv::Mat(modelPoints);
  mOp2 = cv::Mat(eyeModel);

  std::vector<double> rv(3),tv(3),rv2(3),tv2(3);
  mRvec = cv::Mat(rv);
  mRvec2 = cv::Mat(rv2);

  double _d[9] = {1,0,0,
                  0,-1,0,
                  0,0,-1};
  double _d2[9] = {1,0,0,
                   0,-1,0,
                   0,0,-1};

  Rodrigues(cv::Mat(3,3,CV_64FC1,_d),mRvec);
  Rodrigues(cv::Mat(3,3,CV_64FC1,_d2),mRvec2);

  tv[0]=0; tv[1]=0; tv[2]=1;
  mTvec = cv::Mat(tv);

  tv2[0]=0; tv2[1]=0; tv2[2]=1;
  mTvec2 = cv::Mat(tv2);

  camMatrix = cv::Mat(3,3,CV_64FC1);
  camMatrix2 = cv::Mat(3,3,CV_64FC1);

  // Load face detection and pose estimation models.
  detector = dlib::get_frontal_face_detector();
  dlib::deserialize(PoseModelPath) >> pose_model;

  rtree = new CvRTrees;
  rtree->load(GazeTrackerModelPath.c_str());
}

void GazeMatrix::_setupIrisDetector() {
  char syspath[100], pwd[100];
  PyObject *pName, *pModule, *pIn, *pOut;

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
  pProcessEye = PyObject_GetAttrString(pModule, "processEye");
  pProcessEyeArgs = PyTuple_New(0);

  if (pIn == NULL || pOut == NULL || pProcessEye == NULL)
  {
    printf ("Iris finder module not valid\n");
    exit(1);
  }

  pyBufData = (uint8_t *)PyArray_DATA(pIn);
  pyResData = (double *)PyArray_DATA(pOut);
}

void GazeMatrix::processFrame(cv::Mat frame) {
  cv::Mat small, track;

  cv::resize(frame, small, cv::Size(640, 360), 0, 0, cv::INTER_NEAREST);
  dlib::cv_image<dlib::bgr_pixel> csmall(small);
  cv::Rect area;
  cv::Size size;

  area = cv::Rect(0,0,1280,720);
  size = cv::Size(640,360);
  track = small.clone();
  dlib::cv_image<dlib::bgr_pixel> ctrack(track);

  // Detect faces
  std::vector<dlib::rectangle> empty;
  std::vector<dlib::rectangle> faces = detector(ctrack);

  // Find the pose of each face.
  std::vector<dlib::full_object_detection> shapes;

  for (unsigned long i = 0; i < faces.size(); ++i)
    shapes.push_back(pose_model(ctrack, faces[i]));

  _calculateFeatures(frame, area, size, faces, shapes, CV_RGB(0,255,0));
}

void GazeMatrix::_calculateFeatures(
  cv::Mat& image,
  cv::Rect& area,
  cv::Size& size,
  const std::vector<dlib::rectangle>& faces,
  const std::vector<dlib::full_object_detection>& dets,
  const cv::Scalar color
) {
  // Variables
  int interOcDist = -1;
  cv::Mat newImg, src_gray,crop,crop2, eyeMap,leftCrop, rightCrop;
  newImg= image.clone();
  std::vector<cv::Point2f> srcImagePoints, LeftEyePoints, RightEyePoints, LeftEyePointsModel,LeftEllipse, RightEllipse, allKeyPoints;
  std::vector<float> allKeyPointsX, allKeyPointsY;
  cv::Point LeftGaze, RightGaze, RightCorner, LeftCorner, TopCorner, BotCorner, p1, meanLeft, meanRight;
  cv::Rect LeftEye, RightEye, rectLeft, rectRight;
  cv::Point TopLeft, TopRight, BotLeft, BotRight;
  cv::Point RightPupil;
  cv::Point leftPupil;
  TopLeft.x = image.cols/4;
  TopLeft.y = image.rows/4;
  BotLeft.x = image.cols/4;
  BotLeft.y = 3*image.rows/4;
  TopRight.x = 3*image.cols/4;
  TopRight.y = image.rows/4;
  BotRight.x = 3*image.cols/4;
  BotRight.y = 3*image.rows/4;

  const long points[] = {1,16,0,28,30,0,18,21,0,23,26,0,31,35,1,37,41,1,43,47,1,49,59,1,61,67,1,-1};

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

    const dlib::full_object_detection& d = dets[i];

    for(unsigned long i =0; i<=67; i++)
    {
      p1.x = d.part(i).x()*xScale+area.x;
      p1.y = d.part(i).y()*yScale+area.y;
      std::string x = std::to_string(i);
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
          LeftEye.x = p1.x - 20*LeftSize/100;
          LeftEye.y = p1.y - 50*LeftSize/100;
          LeftEye.width = LeftSize;
          LeftEye.height = LeftSize;

          //This prevents the program from crashing if a face is lost
          try
          {
            crop = newImg(LeftEye);
            leftPupil = _findEyeCenter(newImg,LeftEye,"Left Eye",interOcDist);

          }
          catch(...)
          {
            std::cout<<"LEFT BOUNDING BOX EXCEPTION"<<std::endl;
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
          RightEye.x = p1.x - 5*RightSize/100;
          RightEye.y = p1.y - 50*RightSize/100;
          RightEye.width = RightSize;
          RightEye.height = RightSize;

          try{
            crop2 = newImg(RightEye);
            RightPupil = _findEyeCenter(newImg,RightEye,"Right Eye",interOcDist);
          }
          catch(...) {
            std::cout<<"RIGHT EYE EXCEPTION HERE !!!!!!!!!!!!!!!!!!!!"<<std::endl;
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
    rectLeft.x = startX - 10*LeftSize/100;
    rectLeft.y = startY - 20*LeftSize/100;
    lEllipse = fitEllipse(LeftEllipse);
    rectLeft.width = lEllipse.size.height + 10*LeftSize/100;
    rectLeft.height = 3*lEllipse.size.height/4;

    try
    {
      leftCrop = newImg(rectLeft);
      leftPupil = _findEyeCenter(newImg,rectLeft,"Left Eye",interOcDist);
      leftPupil.x += rectLeft.x;
      leftPupil.y += rectLeft.y;
      RightCorner.x = leftPupil.x - 15*LeftSize/100;
      RightCorner.y = leftPupil.y;
      LeftCorner.x = leftPupil.x + 15*LeftSize/100;
      LeftCorner.y = leftPupil.y;
      TopCorner.x = leftPupil.x;
      TopCorner.y = leftPupil.y - 15*LeftSize/100;
      BotCorner.x = leftPupil.x;
      BotCorner.y = leftPupil.y + 15*LeftSize/100;
      LeftEyePointsModel.push_back(leftPupil);
      LeftEyePointsModel.push_back(RightCorner);
      LeftEyePointsModel.push_back(LeftCorner);
      LeftEyePointsModel.push_back(TopCorner);
      LeftEyePointsModel.push_back(BotCorner);
      circle(image,leftPupil,2, cv::Scalar(255,125,255), 2);

    }
    catch(...)
    {
      std::cout<<"LEFT BOUNDING BOX EXCEPTION"<<std::endl;
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

    rectRight.x = right_startX - 10*RightSize/100;
    rectRight.y = right_startY - 20*RightSize/100;
    rEllipse = fitEllipse(RightEllipse);
    rectRight.width = rEllipse.size.height + 10*RightSize/100;
    rectRight.height = 3*rEllipse.size.height/4;

    try
    {
      rightCrop = newImg(rectRight);
      RightPupil = _findEyeCenter(newImg,rectRight,"Left Eye",interOcDist);
      RightPupil.x += rectRight.x;
      RightPupil.y += rectRight.y;
      circle(image,RightPupil,2, cv::Scalar(0,0,255), 2);

    }
    catch(...)
    {
      std::cout<<"Right BOUNDING BOX EXCEPTION"<<std::endl;
    }
  }

  cv::Mat ip, ip2;
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

  double roll, pitch, yaw;
  if(srcImagePoints.size() == 7 && LeftEyePointsModel.size() == 5) {
    _loadWithPoints(
      ip,
      ip2,
      image,
      leftPupil,
      RightPupil,
      LeftEyePoints,
      RightEyePoints,
      LeftEllipse,
      RightEllipse,
      meanLeft,
      meanRight,
      pitch,
      yaw,
      roll,
      interOcDist
    );
  }

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

    std::cout << "gaze matrix: " << gazeMatrix << std::endl;
  }

  std::cout << "result: " << result << std::endl;

  rightCentre.x = (RightEyePoints[0].x + RightEyePoints[2].x)/2;
  rightCentre.y = (RightEyePoints[0].y + RightEyePoints[2].y)/2;
  srcImagePoints.clear();
  LeftEyePointsModel.clear();
  LeftEllipse.clear();
  LeftEyePoints.clear();
  RightEyePoints.clear();
}

//Find's the centre of the Iris from appropriate eye region. This is done by passing the appropriate objects to the Python Iris Finder
cv::Point GazeMatrix::_findEyeCenter(
  cv::Mat face,
  cv::Rect eye,
  std::string debugWindow,
  double interOcDist
) {
  cv::Mat eyeROIUnscaled = face(eye);
  cv::Mat eyeROI (cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/eye.width) * eye.height),CV_8UC3,pyBufData,cv::Mat::AUTO_STEP);
  _scaleToFastSize(eyeROIUnscaled, eyeROI);
  pyResData[0] = (double)eyeROI.rows;
  pyResData[1] = (double)eyeROI.cols;
  pyResData[2] = interOcDist;
  //Iris Finder is called now
  PyObject_CallObject(pProcessEye, pProcessEyeArgs);
  cv::Point result (pyResData[0],pyResData[1]);
  //Rescale the point
  return _unscalePoint(result,eye);
}

// Resuzes Mat object to appropriate size
void GazeMatrix::_scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

//Undoes Scaling
cv::Point GazeMatrix::_unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

//Handles Interfacing with the 3-D model and things such as backproject/ reproject are doen here to get Roll, Pitch, Yaw
// Modifies the given Roll, Pitch, Yaw, interOcDist.
void GazeMatrix::_loadWithPoints(
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
) {
  int max_d = MAX(img.rows,img.cols);
  float delX = 30.0;
  float delY = 30.0;
  float delZ = 30.0;
  int rightCentre, leftCentre;
  std::vector<cv::Point3f> axisPoints, ModelLeft;
  cv::Mat axisMatrix;
  cv::Mat ModelLeftMatrix;

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
  cv::solvePnP(mOp,ip,camMatrix,cv::Mat(1,4,CV_64FC1,_dc),mRvec,mTvec,false,CV_EPNP);

  double rot[9] = {0};
  cv::Mat rotM(3,3,CV_64FC1,rot);

  //Splits the rotation matrix into a vector as well a matrix
  cv::Rodrigues(mRvec,rotM);
  double* _r = rotM.ptr<double>();
  // printf("rotation mat: \n %.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",
  //   _r[0],_r[1],_r[2],_r[3],_r[4],_r[5],_r[6],_r[7],_r[8]);
  // printf("trans vec: \n %.3f %.3f %.3f\n",tv[0],tv[1],tv[2]);

  std::vector<double> tv(3);
  tv[0]=0; tv[1]=0; tv[2]=1;

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
  cv::solvePnP(mOp2,ip2,camMatrix2,cv::Mat(1,4,CV_64FC1,_dc),mRvec2,mTvec2,false,CV_EPNP);

  double rot2[9] = {0};
  cv::Mat rotM2(3,3,CV_64FC1,rot2);
  cv::Rodrigues(mRvec2,rotM2);
  double* _r2 = rotM2.ptr<double>();
  std::vector<double> tv2(3);
  tv2[0]=0; tv2[1]=0; tv2[2]=1;
  double _pm2[12] = {_r2[0],_r2[1],_r2[2],tv2[0],
                     _r2[3],_r2[4],_r2[5],tv2[1],
                     _r2[6],_r2[7],_r2[8],tv2[2]};

  cv::Matx34d P2(_pm2);
  cv::Mat KP2 = camMatrix2 * cv::Mat(P2);

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
  interOcDist = std::abs(leftCentre - rightCentre);
  // cout<<"Inter occular Distance: "<<interOcDist<<endl;
  cv::Point2f leftLeftEdge, leftRightEdge, rightLeftEdge, rightRightEdge; // convention is lower case denotes which eye , upper case denotes left or right edge
  //reproject object points - check validity of found projection matrix
  for (int i=0; i<mOp.rows; i++) {
    cv::Mat_<double> X = (cv::Mat_<double>(4,1) << mOp.at<float>(i,0),mOp.at<float>(i,1),mOp.at<float>(i,2),1.0);
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
  val = 180.0 / std::PI;

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

  float yDiff = std::abs(leftPupil.y - lEllipse.center.y);
  cv::Point2f gazePoint;
  rotM = rotM.t();// transpose to conform with majorness of opengl matrix

}
