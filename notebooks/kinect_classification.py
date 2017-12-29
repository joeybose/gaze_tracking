#!/usr/bin/env python
import pdb
import cv2
import dlib
import numpy as np
import features
import pickle
import scipy
import argparse
import websock
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

setGlobalLogger(None)
parser = argparse.ArgumentParser(description='Live classification of gaze tracker')
parser.add_argument('-m', '--model-file', default='models/classifier.pkl', help='Scikit Learn classifier to load for classification serialized to Jobline')
parser.add_argument('-b', '--averaging-buffer', default=10, help='Size of buffer to smooth classifications', type=int)
args = parser.parse_args()

def gaze_vector(frame):
    global tracker
    global web_comm
    dets = []
    if tracker.get_position().is_empty():
        dets = features.FaceDetection.calculate(frame)
    else:
        rect = tracker.get_position()
        y1 = rect.top()
        y2 = rect.bottom()
        x1 = rect.left()
        x2 = rect.right()
        dets = features.FaceDetection.calculate(frame[y1:y2, x1:x2].copy())

    if not dets:
        # clear the tracker
        tracker = dlib.correlation_tracker()

    # choose largest face
    for det in sorted(dets, key=lambda d: d.area(), reverse=True):
        if tracker.get_position().is_empty():
            tracker.start_track(frame, det)
        else:
            tracker.update(frame)
            tracker_pos = tracker.get_position()
            det = dlib.rectangle(
                int(tracker_pos.left() + det.left()),
                int(tracker_pos.top() + det.top()),
                int(tracker_pos.left() + det.right()),
                int(tracker_pos.top() + det.bottom())
            )

        landmarks = features.FaceLandmarks.calculate(frame, det)
        orientation = features.HeadOrientation.calculate(frame, landmarks)
        eye_roi = features.EyeRegions.calculate(landmarks)
        left_pupil = features.IrisFinder.calculate(frame, eye_roi.left)
        right_pupil = features.IrisFinder.calculate(frame, eye_roi.right)
        ellipses = features.EyeEllipses.calculate(landmarks)

        vector = features.GazeMatrix.calculate(
            landmarks,
            orientation,
            ellipses,
            left_pupil,
            right_pupil
        )

        # Reshape to fix deprecation warning in Scikit Learn 0.17
        return np.array(vector).reshape(1, -1)
    return None
clf = joblib.load(args.model_file)
try:
    from pylibfreenect2 import OpenCLPacketPipeline
    pipeline = OpenCLPacketPipeline()
except:
    from pylibfreenect2 import CpuPacketPipeline
    pipeline = CpuPacketPipeline()

enable_rgb = True
enable_depth = False

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)
serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0 
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)
undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)
# Register listeners
device.setColorFrameListener(listener)
device.start()

dots = [
  {
    "x": 0.15,
    "y": 0.15,
    "color": [0, 0, 255],
    "size": 20,
    "label": "1"
  },
  {
    "x": 0.85,
    "y": 0.15,
    "color": [0, 0, 255],
    "size": 20,
    "label": "2"
  },
  {
    "x": 0.15,
    "y": 0.85,
    "color": [0, 0, 255],
    "size": 20,
    "label": "3"
  },
  {
    "x": 0.85,
    "y": 0.85,
    "color": [0, 0, 255],
    "size": 20,
    "label": "4"
  }
]

tracker = dlib.correlation_tracker()
max_classifications = args.averaging_buffer
classifications_cached = []
last_certain_classification = None

while True:
    frames = listener.waitForNewFrame()
    if enable_depth:
        ir = frames["ir"]
        depth = frames["depth"]
    if enable_rgb:
	color = frames["color"].asarray()
	frame = color[:,:, :3].copy()
        cv2.flip(frame, 1, dst=frame)
	preview = frame.copy()
        vector = gaze_vector(frame)
        if vector is not None:
            prediction = clf.predict(vector)

            if len(classifications_cached) < max_classifications:
                classifications_cached.append(prediction)
            else:
                last_certain_classification = scipy.stats.mode(classifications_cached)
                classifications_cached = []
            if last_certain_classification != None:
                dot = dots[last_certain_classification[0]]
                point = (int(preview.shape[1] * dot['x']), int(preview.shape[0] * dot['y']))
                color = (dot['color'][0], dot['color'][1], dot['color'][2])
                cv2.circle(preview, point, dot['size'], color, -1)
        rect = tracker.get_position()
        if not rect.is_empty():
            cv2.rectangle(
                preview,
                (int(rect.left()), int(rect.top())),
                (int(rect.right()), int(rect.bottom())),
                (0, 0, 255),
                1
            )
	cv2.imshow("color", cv2.resize(preview,(int(1920 / 3), int(1080 / 3))))
	listener.release(frames)
    cv2.waitKey(1)
    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break
device.stop()
device.close() 
sys.exit(0)
