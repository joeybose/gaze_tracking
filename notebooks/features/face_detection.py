from feature import Feature
import dlib
import cv2

class FaceDetection(Feature):
    detector = dlib.get_frontal_face_detector()

    @classmethod
    def calculate(cls, input):
        dets = cls.detector(input)
        return dets

    @classmethod
    def show(cls, input, detections):
        # OpenCV drawing functions directly edit the image. Make a copy to
        # preserve the original.
        img = input.copy()

        for det in detections:
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0, 0), 3)

        return img
