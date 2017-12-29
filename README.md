# Gaze Tracker

## Python Dependencies
```
brew install python cmake boost boost-python pkg-config
brew tap homebrew/science
brew install opencv --with-contrib
pip install -r requirements.txt
under notebooks/ one can run live_classification.py
please note code has been optimized to not display results window
but a quick cv2.imshow can be used to remedy this.
```

## C++ Dependencies
```
brew install tclap pkg-config glfw
```

## OSX Yosemite and greater.
OSX does not ship with X11 starting with Yosemite. So there is an extra step of installing X11 and configuring it so dlib in both the python and C++ versions of this project can find it.
1. Download and Install XQuartz from https://support.apple.com/en-ca/HT201341
2. After installation X11 is stored in /opt/X11/ so it's necessary to make a symlink to /usr/local/include/X11/ (or wherever your OS looks for X11)

```
(sudo) ln -s /opt/X11/include/X11/ /usr/local/include/X11
```
Clone dlib from https://github.com/davisking/dlib

Download shape_predictor_68_face_landmarks.dat from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

### dlib Setup

1. Symlink the dlib directory to gaze-tracker
2. Copy shape_predictor_68_face_landmarks.dat (download online) to the dlib folder

### Hints
* If you can't find numpy, add the numpy include path to PYTHON_PATH
* If you can't find glfw, add the pc file path to PKG_CONFIG_PATH
* If you can't find libraries during compile time, add the library path to LD_PATH
```
export LIBRARY_PATH=/usr/local/lib/
```
* If you can't find libraries during runtime(dyld), add the library path to DYLD_LIBRARY_PATH
* Refer to this document for further explanations and build instructions/hints
https://docs.google.com/document/d/1_Ao_8QlBQtomENin-cXa0mKOuQ802slcMpCLzWxz29s

## Building C++

```
mkdir build && cd build
cmake ..
make
```

## Usage

Run `ml_gaze_tracker_1 -h` to see a list of command line arguments.

```
   ./ml_gaze_tracker_1  [-e <string>] [-r <string>] [-l] [--] [--version]
                        [-h]


Where:

   -e <string>,  --test-file <string>
     File to test on

   -r <string>,  --train-file <string>
     File to train on.

   -l,  --load-model
     Loads model instead of training. -l, -r, -e must all be specified
     together to load model

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.


   Gaze Tracker
```

## Training

In order to start the training process, data should be collected. To run the application with data collection, use the --train-file argument:

```
./ml_gaze_tracker_1 --train-file <path/to/train.file>
```

To start collection on a particular corner, press the corresponding key in *keyboard controls*. The training data will be appended to the file specified with `--train-file`.

It is recommended to train for 10 seconds at a time for a given head pose and eye orientation. Different head poses and eye orientations are also suggested.

Once the data collection is complete, the application should be restarted with the `--load-model` flag. Note: the `--test-file` is required as well.

```
./ml_gaze_tracker_1 --load-model --train-file <path/to/train.file> --test-file <path/to/test.file>
```

### Keyboard Controls
* q: Start collection of top left
* w: Start collection of top right
* a: Start collection of bottom left
* s: Start collection of bottom right
