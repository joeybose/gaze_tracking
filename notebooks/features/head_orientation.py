from feature import Feature
from types import Orientation
import dlib
import cv2
import math
import numpy as np

class HeadOrientation(Feature):
    model_points = np.matrix([
        (36.8301,78.3185,52.0345),      # nose (v 1879)
        (-12.6448,108.891,11.8577),     # Left edge Left eye
        (18.6566,106.811,18.9713),      # Right edge Left eye
        (54.0573,106.811,18.4686),      # Left edge Right eye
        (85.1435,108.891,10.4489),      # Right edge RIght eye
        (14.8498,51.0115,30.2378),      # l mouth (v 1502)
        (58.1825,51.0115,29.6224)       # r mouth (v 695)
    ])

    dist_coeffs = (0,0,0,0)

    @classmethod
    def calculate(cls, img, landmarks):
        max_d = max(img.shape[0], img.shape[1])
        x = landmarks.rect.left()
        y = landmarks.rect.top()

        image_points = np.array(
            map(
                lambda i: (float(landmarks.part(i).x), float(landmarks.part(i).y)),
                (30,36,39,42,45,48,54)
            )
        )

        camera_matrix = np.matrix([
            (max_d, 0, img.shape[1]/2.0),
            (0, max_d, img.shape[0]/2.0),
            (0, 0, 1.0)
        ])

        (retval, rvec, tvec) = cv2.solvePnP(
            cls.model_points,
            image_points,
            camera_matrix,
            cls.dist_coeffs,
            # flags=cv2.CV_EPNP
        )
        rot_m = cv2.Rodrigues(rvec)[0]

        val = 180.0 /  math.pi;

        # Standard roll pitch yaw calculations from rotation matrix
        yaw = math.atan2(rot_m[1][0], rot_m[0][0]) * val;
        pitch = math.atan2(-1*rot_m[2][0],math.sqrt(rot_m[2][1]*rot_m[2][1] + rot_m[2][2]*rot_m[2][2])) * val;
        roll = math.atan(rot_m[2][1]/rot_m[2][2]) * val;

        return Orientation(yaw=yaw, pitch=pitch, roll=roll)

    @classmethod
    def show(cls, input, detections):
        # OpenCV drawing functions directly edit the image. Make a copy to
        # preserve the original.
        img = input.copy()
        return img
