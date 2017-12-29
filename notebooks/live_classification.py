#!/usr/bin/env python

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
from websock import WSHandler
from multiprocessing import Process, Pipe

parser = argparse.ArgumentParser(description='Live classification of gaze tracker')
parser.add_argument('-m', '--model-file', default='models/classifier.pkl', help='Scikit Learn classifier to load for classification serialized to Jobline')
parser.add_argument('-b', '--averaging-buffer', default=10, help='Size of buffer to smooth classifications', type=int)
parser.add_argument('-c', '--capture-device', help='Index of capture device', type=int, default=0)
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

cap = cv2.VideoCapture(args.capture_device)
clf = joblib.load(args.model_file)
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
parent_conn, child_conn = Pipe()
p = Process(target=websock.setup, args=(child_conn,))
p.start()
max_classifications = args.averaging_buffer
classifications_cached = []
last_certain_classification = None

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
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
                #print dot['label']
                parent_conn.send(dot['label']) 
                point = (int(preview.shape[1] * dot['x']), int(preview.shape[0] * dot['y']))
                color = (dot['color'][0], dot['color'][1], dot['color'][2])
                cv2.circle(preview, point, dot['size'], color, -1)
        else:
            parent_conn.send("-1")

        # draw tracking bounding box
        rect = tracker.get_position()
        if not rect.is_empty():
            cv2.rectangle(
                preview,
                (int(rect.left()), int(rect.top())),
                (int(rect.right()), int(rect.bottom())),
                (0, 0, 255),
                1
            )

        cv2.imshow('frame', preview)

    cv2.waitKey(1)