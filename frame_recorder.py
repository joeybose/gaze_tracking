#!/usr/bin/env python

import argparse
import logging
import sys
import os
import numpy as np
import cv2
import json
import cv_constants as cv_const
import time

parser = argparse.ArgumentParser(description='Records frames')
parser.add_argument('-v', '--verbose', action='store_true', default=False)
parser.add_argument('folder', default='output', help='Folder to save frames to')
parser.add_argument('-c', '--capture-device', help='Index of capture device', type=int, default=0)
parser.add_argument('-w', '--width', help='Width of video stream', type=int, default=640)
parser.add_argument('-H', '--height', help='Height of video stream', type=int, default=480)
parser.add_argument('-f', '--fps', help='FPS to record at', type=int, default=20)
parser.add_argument('-F', '--flip', help='Flip image horizontally', action='store_true', default=False)
parser.add_argument('-a', '--save-all', help='FPS to record at', action='store_true', default=False)
parser.add_argument('-p', '--prefix', default='frame', help='Prefix of file name')
parser.add_argument('-l', '--labels', default=None, help='JSON mapping of keys to labels')
parser.add_argument('-t', '--file-type', default='jpg', help='File type', choices=['jpg', 'png'])
parser.add_argument('-q', '--file-quality', default=95, type=int, choices=range(0, 101, 5), help='Quality of saved file. Applicible only to JPEG format')
parser.add_argument('-C', '--file-compression', default=3, type=int, choices=range(0, 10), help='Compression level of saved file. Applicible only to PNG format')
parser.add_argument('-T', '--with-timestamps', action='store_true', default=False, help='Replaces indexes with timestamps in the file names')
parser.add_argument('-d', '--dots', default=None, help='Locations of dots JSON file')
args = parser.parse_args()

level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(stream=sys.stdout, level=level)

logging.debug("Using OpenCV version: {}".format(cv2.__version__))

def frame_file_name(folder, prefix, index, file_type, with_timestamp, label=None):
    if with_timestamp:
        index = time.time()

    if label:
        frame_path = os.path.join(folder, '{}_{}_{}.{}')
        return frame_path.format(prefix, index, label, file_type)
    else:
        frame_path = os.path.join(folder, '{}_{}.{}')
        return frame_path.format(prefix, index, file_type)

logging.info('Frame output folder: {}'.format(args.folder))
if not os.path.exists(args.folder):
    logging.info('Output folder ({}) does not exist. Creating...'.format(args.folder))
    os.makedirs(args.folder)

example_frame_path = frame_file_name(args.folder, args.prefix, '<index>', args.file_type, args.with_timestamps)
logging.info('Frames saved to: {}'.format(example_frame_path))

if args.save_all:
    logging.info('Saving all frames')
else:
    logging.info('Saving only labeled frames')

labels = {}
if args.labels:
    logging.info('Loading labels file: {}'.format(args.labels))
    f = open(args.labels, 'r')
    data = json.loads(f.read())
    if isinstance(data, dict):
        labels = data
        logging.info('Loaded labels file successfully')
    else:
        logging.error('Labels JSON contents does not contain a dictionary')
else:
    logging.info('No labels file specified')

dots = []
if args.dots:
    logging.info('Loading dots file: {}'.format(args.labels))
    f = open(args.dots, 'r')
    data = json.loads(f.read())
    if isinstance(data, list):
        dots = data
        logging.info('Loaded dots file successfully')
    else:
        logging.error('Dots JSON contents does not contain an array')
else:
    logging.info('No dots file specified')

file_options = None
if args.file_type == 'jpg':
    file_options = (cv_const.IMWRITE_JPEG_QUALITY, args.file_quality)
    logging.info('Saving in JPEG format with quality: {}'.format(args.file_quality))
elif args.file_type == 'png':
    file_options = (cv_const.IMWRITE_PNG_COMPRESSION, args.file_compression)
    logging.info('Saving in PNG format with compression: {}'.format(args.file_compression))

logging.info('Capturing on device {} ({}x{}) at {} fps'.format(args.capture_device, args.width, args.height, args.fps))
cap = cv2.VideoCapture(args.capture_device)
cap.set(cv_const.CAP_PROP_FRAME_WIDTH,args.width);
cap.set(cv_const.CAP_PROP_FRAME_HEIGHT,args.height);

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    if args.flip:
        cv2.flip(frame, 1, dst=frame)

    preview = frame.copy()

    if args.dots:
        for dot in dots:
            point = (int(preview.shape[1] * dot['x']), int(preview.shape[0] * dot['y']))
            color = (dot['color'][0], dot['color'][1], dot['color'][2])
            cv2.circle(preview, point, dot['size'], color, -1)

            if 'label' in dot:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(preview, dot['label'],point, font, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('frame', preview)

    key_code = cv2.waitKey(1000/args.fps)

    if key_code == 27: # esc key
        break
    if key_code != -1 and chr(key_code) in labels:
        label = labels[chr(key_code)]
        cv2.imwrite(frame_file_name(args.folder, args.prefix, i, args.file_type, args.with_timestamps, label), frame, file_options)
    elif args.save_all:
        cv2.imwrite(frame_file_name(args.folder, args.prefix, i, args.file_type, args.with_timestamps), frame, file_options)

    i += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
