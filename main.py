import cv2
import sys
import os
import random

import numpy as np

from statistics import mean
# from mpmath import *
from math import sqrt
from time import sleep
from tqdm import tqdm
from tkinter.filedialog import askopenfilename

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
print(f"OpenCV v{major_ver}.{minor_ver}.{subminor_ver}")


# iv.dps = 10
def NCDist(boxA, boxB):
    centerCoordA = (boxA[0] + (boxA[2] / 2), boxA[1] + (boxA[3] / 2))
    centerCoordB = (boxB[0] + (boxB[2] / 2), boxB[1] + (boxB[3] / 2))
    centerCoordA = (centerCoordA[0] / boxB[2], centerCoordA[1] / boxB[3])
    centerCoordB = (centerCoordB[0] / boxB[2], centerCoordB[1] / boxB[3])
    return sqrt((centerCoordA[0] - centerCoordB[0]) ** 2 + (centerCoordA[1] - centerCoordB[1]) ** 2)


def CDist(boxA, boxB):
    centerCoordA = (boxA[0] + (boxA[2] / 2), boxA[1] + (boxA[3] / 2))
    centerCoordB = (boxB[0] + (boxB[2] / 2), boxB[1] + (boxB[3] / 2))
    return sqrt((centerCoordA[0] - centerCoordB[0]) ** 2 + (centerCoordA[1] - centerCoordB[1]) ** 2)


def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# Set up tracker.


def Analise(videos, vid_folder, tracker_t):
    ious = [0.0] * 10
    fpss = [0.0] * 10
    dsts = [0.0] * 10
    ndsts = [0.0] * 10
    for j, seq_ID in videos:
        frames_folder = os.path.join(vid_folder, "frames", seq_ID)
        anno_file = os.path.join(vid_folder, "anno", seq_ID + ".txt")
        anno = np.loadtxt(anno_file, delimiter=",")
        frames_list = [frame for frame in os.listdir(frames_folder) if
                       frame.endswith(".jpg")]
        if not len(anno) == len(frames_list):
            print("Not the same number of frames and annotation!")
            continue
        # Define an initial bounding box
        bbox = (anno[0][0], anno[0][1], anno[0][2], anno[0][3])
        frame = cv2.imread(os.path.join(frames_folder, "0.jpg"))
        if tracker_t == "MOSSE":
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_t == "TLD":
            tracker = cv2.TrackerTLD_create()
        elif tracker_t == "GOTURN":
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_t == "BOOSTING":
            tracker = cv2.TrackerBoosting_create()
        elif tracker_t == "MIL":
            tracker = cv2.TrackerMIL_create()
        elif tracker_t == "KCF":
            tracker = cv2.TrackerKCF_create()
        elif tracker_t == "MEDIANFLOW":
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_t == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        ok = tracker.init(frame, bbox)
        if not ok:
            print("Initialisation error")
            continue
        for i in range(len(frames_list)):
            frame_file = str(i) + ".jpg"
            imgs_file = os.path.join(frames_folder, frame_file)

            frame = cv2.imread(imgs_file)
            anno_bbox = (anno[i][0], anno[i][1], anno[i][2], anno[i][3])
            x_min = int(anno[i][0])
            x_max = int(anno[i][2] + anno[i][0])
            y_min = int(anno[i][1])
            y_max = int(anno[i][3] + anno[i][1])

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2, 1)

            # Start timer
            timer = cv2.getTickCount()
            # Update tracker
            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                # print("Tracking failure detected")
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                            2)

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            cv2.imshow("Tracking", frame)
            ious[j] += IOU(bbox, anno_bbox)
            dsts[j] += CDist(bbox, anno_bbox)
            ndsts[j] += NCDist(bbox, anno_bbox)
            fpss[j] += fps
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                sys.exit()
        ious[j] /= len(frames_list)
        fpss[j] /= len(frames_list)
        dsts[j] /= len(frames_list)
        ndsts[j] /= len(frames_list)
        print(f" IOU = {ious[j]}, FPS = {fpss[j]}, CDist = {dsts[j]}, NCDist = {ndsts[j]}")
        # cv2.imwrite(anno_file, frame)
    return ious, fpss, dsts, ndsts


tracker_type = "CSRT"
chunk_folder = "F:\\Torents\\TRAIN_0".upper()
vid = random.choices(os.listdir(os.path.join(chunk_folder, "frames")), k=10)
vid = tqdm(enumerate(vid))
iou, fps, dst, ndst = Analise(vid, chunk_folder, tracker_type)
print(f"{tracker_type} tracker : Average IOU = {mean(iou)}, average FPS = {mean(fps)},",
      f"average CDist = {mean(dst)}, average NCDist = {mean(ndst)}, score = {mean(iou) * mean(fps) / mean(dst)}")
