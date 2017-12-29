# Gaze Tracker Advertisement Demo

## Setup

```
bower install
```

## Running

### Run the live classifier script

```
cd ../notebooks
./live_classification.py --model models/GBT.pkl -b 7 -c <camera_index>
```

Set *<camera_index>* to a number to select the webcam to use. Leave out the flag to use the default webcam.


### Open Client

```
open index.html
```
