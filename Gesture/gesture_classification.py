#!/usr/bin/env python
import pdb
from msvcrt import getch
import cv2
from keras.models import model_from_json
import numpy as np
import pickle
import scipy
import argparse
import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.regularizers import l2, activity_l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils.np_utils import probas_to_classes
from keras import backend as K
from PIL import Image as pil_image
parser = argparse.ArgumentParser(description='Live classification of gaze tracker')
parser.add_argument('-b', '--averaging-buffer', default=10, help='Size of buffer to smooth classifications', type=int)
parser.add_argument('-c', '--capture-device', help='Index of capture device', type=int, default=0)
args = parser.parse_args()
model_file_prefix = 'hand_rgb'
cap = cv2.VideoCapture(args.capture_device)
max_classifications = args.averaging_buffer
classifications_cached = []
json_file = open('hand_rgb.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("googlenet_handjoey_weights.04-9.19.hdf5")
last_certain_classification = None
loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
font = cv2.FONT_HERSHEY_SIMPLEX
#fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
#out = cv2.VideoWriter('gesture.mp4',fourcc,20.0,(640,480))
try:
	while(cap.isOpened()):
		#key = ord(getch())
		# ESC key
		#if key == 27:
		#	break
		ret, frame = cap.read()
		if ret:
			cv2.flip(frame, 1, dst=frame)
			preview = frame.copy()
			preview = scipy.misc.imresize(preview.astype(np.float32), size=(299, 299), interp='bilinear')
			preview = np.expand_dims(preview, axis=0)
			preview = preview.transpose((0, 3, 1, 2))
			preview = preview / 255.0
			probs = loaded_model.predict(preview)
			prediction = probas_to_classes(probs) + 1
			print(prediction[0])
			cv2.putText(frame,str(prediction[0]),(10,400), font, 5, (128,220,220),3)
			#out.write(frame)
		if len(classifications_cached) < max_classifications:
			classifications_cached.append(prediction)
		else:
			last_certain_classification = scipy.stats.mode(classifications_cached)
			classifications_cached = []
		cv2.imshow('frame', frame)
		cv2.waitKey(1)
	cv2.destroyAllWindows()
except KeyboardInterrupt:
	pass

