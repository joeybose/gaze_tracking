from feature import Feature
from types import Point,EyeData
import more_itertools as mit
import math
import dlib
import cv2
import numpy as np

class GazeMatrix(Feature):
    @classmethod
    def calculate(
        cls,
        landmarks,
        head_orientation,
        ellipses,
        left_pupil,
        right_pupil,
    ):
        l_parts = landmarks.parts()
        l_x_parts = map(lambda p: p.x, l_parts)
        l_y_parts = map(lambda p: p.y, l_parts)

        # calculate statistics based on landmark locations
        mean_x = np.mean(l_x_parts)
        stdev_x = np.std(l_x_parts)

        mean_y = np.mean(l_y_parts)
        stdev_y = np.std(l_y_parts)

        result = []

        for el in ellipses:
            el_mean_x = np.mean(map(lambda p: p.x, el))
            el_mean_y = np.mean(map(lambda p: p.y, el))

            result.append((el_mean_x - mean_x)/stdev_x)
            result.append((el_mean_y - mean_y)/stdev_y)

            for p in el:
                result.append((p.x - mean_x)/stdev_x)
                result.append((p.y - mean_y)/stdev_y)

        result.append(head_orientation.roll)
        result.append(head_orientation.pitch)
        result.append(head_orientation.yaw)

        eye_centers = EyeData(*map(cls._ellipse_to_center, ellipses))

        left_pupil_displacement = Point(
            left_pupil[0].x - eye_centers.left.x,
            left_pupil[0].y - eye_centers.left.y
        )
        right_pupil_displacement = Point(
            right_pupil[0].x - eye_centers.right.x,
            right_pupil[0].y - eye_centers.right.y
        )

        result.append(left_pupil_displacement.x/stdev_x)
        result.append(left_pupil_displacement.y/stdev_y)
        result.append(right_pupil_displacement.x/stdev_x)
        result.append(right_pupil_displacement.y/stdev_y)

        result.append((left_pupil[0].x - mean_x)/stdev_x)
        result.append((left_pupil[0].y - mean_y)/stdev_y)
        result.append((right_pupil[0].x - mean_x)/stdev_x)
        result.append((right_pupil[0].y - mean_y)/stdev_y)

        result.append((eye_centers.left.x - mean_x)/stdev_x)
        result.append((eye_centers.left.y - mean_y)/stdev_y)
        result.append((eye_centers.right.x - mean_x)/stdev_x)
        result.append((eye_centers.right.y - mean_y)/stdev_y)

        result.append(mean_x)
        result.append(mean_y)
        result.append(stdev_x)
        result.append(stdev_y)

        return result

    @classmethod
    def show(cls, input, gaze_matrix):
        # OpenCV drawing functions directly edit the image. Make a copy to
        # preserve the original.
        img = input.copy()
        return img

    @classmethod
    def _ellipse_to_center(cls, ellipse):
        return Point(
            (ellipse[3].x - ellipse[0].x)/2 + ellipse[0].x,
            (ellipse[3].y - ellipse[0].y)/2 + ellipse[0].y
        )
