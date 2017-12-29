from feature import Feature
from types import EyeData,Point
from eye_ellipses import EyeEllipses
import dlib
import cv2

class EyeRegions(Feature):
    @classmethod
    def calculate(cls, landmarks):
        ellipses = EyeEllipses.calculate(landmarks)

        def ellipseToRect(el):
            # x1, y2 coorespond to top-left corner of corner of bounding box
            # x2, y2 are the bottom-right corner
            x1 = el[0].x
            y1 = el[1].y
            x2 = el[3].x
            y2 = el[5].y

            w = x2 - x1
            h = y2 - y1
            x1 -= 5*w/100
            x2 += 10*w/100
            y1 -= h/2
            y2 += h

            return (Point(x1,y1),Point(x2,y2))

        return EyeData(*map(ellipseToRect, ellipses))

    @classmethod
    def show(cls, input, regions):
        w = regions.right[1].x - regions.left[0].x
        x1 = int(regions.left[0].x - w/5.0)
        x2 = int(regions.right[1].x + w/5.0)

        heightest_y = min(
            regions.left[0].y,
            regions.right[0].y
        )
        lowest_y = max(
            regions.left[1].y,
            regions.right[1].y
        )
        h = lowest_y - heightest_y
        y1 = int(heightest_y - h/5.0)
        y2 = int(lowest_y + h/5.0)

        # OpenCV drawing functions directly edit the image. Make a copy to
        # preserve the original.
        img = input[y1:y2, x1:x2].copy()
        for region in regions:
            cv2.rectangle(
                img,
                (region[0].x-x1, region[0].y-y1),
                (region[1].x-x1, region[1].y-y1),
                (255, 0, 0),
                1
            )

        return img
