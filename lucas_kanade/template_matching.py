import numpy as np
import cv2 as cv
import argparse
import copy
import time
from algorithms import match_template, ssd, sad, ncc

parser = argparse.ArgumentParser(description='This sample demonstrates the template match algorithm.')
parser.add_argument('image', type=str, help='folder of image')
parser.add_argument('algo', type=str, help='SSD|NCC|SAD', nargs='?', default='SSD')

args = parser.parse_args()

print(args.algo)

if args.algo == 'NCC':
    algo = ncc
elif args.algo == 'SAD':
    algo = sad
else:
    algo = ssd


def get_frame(line):
    try:
        tb = [int(i) for i in line.split(',')]
    except:
        tb = [int(i) for i in line.split()]
    return tb


# Reading the data
cap = cv.VideoCapture(args.image + "/img/%04d.jpg")
file1 = open(args.image + '/groundtruth_rect.txt', 'r')
Lines = file1.readlines()
count = 0

while (1):
    ret, frame = cap.read()
    if ret == True:
        # default track the ground truth
        # setting initial tracking window
        x, y, w, h = get_frame(Lines[count])
        track_window = (x, y, w, h)
        roi = copy.deepcopy(frame[y:y + h, x:x + w])

        matched = match_template(frame, roi, algo)
        if args.algo == 'ncc':
            ind = np.unravel_index(np.argmax(matched, axis=None), matched.shape)
        else:
            ind = np.unravel_index(np.argmin(matched, axis=None), matched.shape)
        top_left = ind[::-1]
        bottom_right = (top_left[0] + w, top_left[1] + h)

        frame = cv.rectangle(frame, top_left, bottom_right, [0, 255, 0], 2)

        cv.imshow("Frame", frame)
        cv.imshow("Template to match", roi)

        count += 1

        # in case we want to match custom template
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        # show the output frame
        cv.imshow("Frame", frame)

    else:
        break
