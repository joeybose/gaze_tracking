import sys
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
import net2 as net
import dataset
np.random.seed(4)

n = 224
batch_size = 128
nb_epoch = 5
nb_phase_two_epoch = 5
DATA_TRAIN = '/cortex1/joey/kinect_leap_dataset/Hand_rgb_train.csv'
DATA_VAL = '/cortex1/joey/kinect_leap_dataset/Hand_rgb_test.csv'
# Use heavy augmentation if you plan to use the model with the
# accompanying webcam.py app, because webcam data is quite different from photos.
heavy_augmentation = True

model_file_prefix = 'hand_rgb'

print "loading dataset"

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def load_data(data_train, data_val):
    Inputs_train = []
    Inputs_val = []
    df_train = read_csv(data_train)  # load pandas dataframe
    df_val = read_csv(data_val)
    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    #print(df.count())  # prints the number of values for each column
    df_train = df_train.dropna()  # drop all rows that have missing values in them
    df_val = df_val.dropna()
    print("Starting")
    X_train = df_train['imgPath'].values
    y_train = df_train['label'].values.astype(int)
    y_train = y_train - 1
    X_val = df_val['imgPath'].values
    y_val = df_val['label'].values.astype(int)
    y_val = y_val - 1
    nb_train_samples = len(X_train)
    nb_validation_samples = len(X_val)
    nb_epoch = 50
    #weights = {0:1, 1:1.0835469, 2:3.02, 3:5.2068965, 4:1.2651107, 5:23.2307692, 6:111.262631579, 7:44.0416667}

    for im in X_train:
        img = load_img(im,target_size=(299,299,3))
        x = img_to_array(img)
        #x = x.reshape((1,) + x.shape)
        Inputs_train.append(x)
    train = np.asarray(Inputs_train)

    for im in X_val:
        img = load_img(im, target_size=(299,299,3))
        x = img_to_array(img)
        # x = x.reshape((1,) + x.shape)
        Inputs_val.append(x)
    val = np.asarray(Inputs_val)

    return train,y_train,val,y_val

X, y, Xv, yv = load_data(DATA_TRAIN, DATA_VAL)
tags = ['0','1','2','3','4','5','6','7','8','9']
nb_classes = len(tags)
X = X.astype('float32')
Xv = Xv.astype('float32')
X = X / 255.0
Xv = Xv / 255.0

X_train = X
y_train = y
Y_train = np_utils.to_categorical(y_train, nb_classes)
X_test  = Xv
y_test  = yv
Y_test = np_utils.to_categorical(y_test, nb_classes)

if heavy_augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.5,
        channel_shift_range=0.5,
        fill_mode='nearest')
else:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

datagen.fit(X_train)

def evaluate(model, vis_filename=None):
    Y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(Y_pred, axis=1)

    accuracy = float(np.sum(y_test==y_pred)) / len(y_test)
    print "accuracy:", accuracy

    confusion = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    for (predicted_index, actual_index, image) in zip(y_pred, y_test, X_test):
        confusion[predicted_index, actual_index] += 1

    print "rows are predicted classes, columns are actual classes"
    for predicted_index, predicted_tag in enumerate(tags):
        print predicted_tag[:7],
        for actual_index, actual_tag in enumerate(tags):
            print "\t%d" % confusion[predicted_index, actual_index],
        print
    if vis_filename is not None:
        bucket_size = 10
        image_size = n // 4 # right now that's 56
        vis_image_size = nb_classes * image_size * bucket_size
        vis_image = 255 * np.ones((vis_image_size, vis_image_size, 3), dtype='uint8')
        example_counts = defaultdict(int)
        for (predicted_tag, actual_tag, normalized_image) in zip(y_pred, y_test, X_test):
            example_count = example_counts[(predicted_tag, actual_tag)]
            if example_count >= bucket_size**2:
                continue
            image = dataset.reverse_preprocess_input(normalized_image)
            image = image.transpose((1, 2, 0))
            image = scipy.misc.imresize(image, (image_size, image_size)).astype(np.uint8)
            tilepos_x = bucket_size * predicted_tag
            tilepos_y = bucket_size * actual_tag
            tilepos_x += example_count % bucket_size
            tilepos_y += example_count // bucket_size
            pos_x, pos_y = tilepos_x * image_size, tilepos_y * image_size
            vis_image[pos_y:pos_y+image_size, pos_x:pos_x+image_size, :] = image
            example_counts[(predicted_tag, actual_tag)] += 1
        vis_image[::image_size * bucket_size, :] = 0
        vis_image[:, ::image_size * bucket_size] = 0
        scipy.misc.imsave(vis_filename, vis_image)

print "loading original inception model"

model = net.build_model(nb_classes)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

# train the model on the new data for a few epochs

print "training the newly added dense layers"

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                samples_per_epoch=X_train.shape[0],
                nb_epoch=nb_epoch,
                validation_data=datagen.flow(X_test, Y_test, batch_size=batch_size),
                nb_val_samples=X_test.shape[0],
            )

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.
callback_path = 'output/googlenet_handjoey_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.01, verbose=1),
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint(callback_path, monitor='val_loss', save_best_only=True, verbose=0),
]
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
lrate = 0.001
decay = lrate/10.0
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

print "fine-tuning top 2 inception blocks alongside the top dense layers"

for i in range(1,30):
    print "mega-epoch %d/24" % i
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_phase_two_epoch,
            validation_data=datagen.flow(X_test, Y_test, batch_size=batch_size),
            nb_val_samples=X_test.shape[0],
            callbacks=callbacks)

    evaluate(model, str(i).zfill(3)+".png")
# serialize model to JSON
model_json = model.to_json()
with open("handrgb.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("handrgb.h5")
print("Saved model to disk")
net.save(model, tags, model_file_prefix)
