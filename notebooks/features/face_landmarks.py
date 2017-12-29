from feature import Feature
import dlib
import cv2

class FaceLandmarks(Feature):
    predictor = dlib.shape_predictor('../src/models/shape_predictor_68_face_landmarks.dat')

    @classmethod
    def calculate(cls, img, face_detections):
        # extract object detections to vector
        return cls.predictor(img, face_detections)

    @classmethod
    def show(cls, input, landmarks):
        # OpenCV drawing functions directly edit the image. Make a copy to
        # preserve the original.
        img = input.copy()

        # crop image to face region
        y = landmarks.rect.top()
        w = landmarks.rect.width()
        x = landmarks.rect.left()
        h = landmarks.rect.height()
        img = img[y:y+h, x:x+w]

        # draw points of landmarks
        for mark in landmarks.parts():
            cv2.circle(img,(mark.x - x, mark.y - y), 2, (0,0,255), -1)

        return img
