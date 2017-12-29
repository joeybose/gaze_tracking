from feature import Feature
from types import EyeData
import dlib
import cv2

class EyeEllipses(Feature):
    @classmethod
    def calculate(cls, landmarks):
        return EyeData(
            left=tuple(landmarks.part(i) for i in xrange(36,42)),
            right=tuple(landmarks.part(i) for i in xrange(42,48))
        )

    @classmethod
    def show(cls, input, eye_ellipses):
        x1 = eye_ellipses.left[0].x
        x2 = eye_ellipses.right[3].x

        heightest_y = min(
            eye_ellipses.left[1].y,
            eye_ellipses.left[2].y,
            eye_ellipses.right[1].y,
            eye_ellipses.right[2].y
        )
        lowest_y = max(
            eye_ellipses.left[4].y,
            eye_ellipses.left[5].y,
            eye_ellipses.right[4].y,
            eye_ellipses.right[5].y
        )
        h = lowest_y - heightest_y
        y1 = int(heightest_y - h/5.0)
        y2 = int(lowest_y + h/5.0)

        # OpenCV drawing functions directly edit the image. Make a copy to
        # preserve the original.
        img = input[y1:y2, x1:x2].copy()
        for eye in eye_ellipses:
            for mark in eye:
                cv2.circle(img,(mark.x - x1, mark.y - y1), 2, (0,0,255), -1)

        return img
