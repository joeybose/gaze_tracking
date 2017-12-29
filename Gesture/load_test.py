import sys
from keras.models import model_from_json
import json
import pdb
import numpy as np
from collections import defaultdict

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
np.random.seed(4)

n = 224
batch_size = 128
nb_epoch = 5
nb_phase_two_epoch = 5
DATA_TRAIN = '/cortex1/joey/kinect_leap_dataset/Hand_rgb_train.csv'
DATA_VAL = '/cortex1/joey/kinect_leap_dataset/Hand_rgb_test.csv'
# Use heavy augmentation if you plan to use the model with the
# accompanying webcam.py app, because webcam data is quite different from photos.
json_file = open('hand_rgb.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("googlenet_handjoey_weights.04-9.19.hdf5")
pdb.set_trace()
print("Loaded model from disk")
