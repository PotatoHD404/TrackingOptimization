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


# pandas frame
# iv.dps = 10
def NCDist(T, GT):
    centerCoordT = (T[0] + (T[2] / 2), T[1] + (T[3] / 2))
    centerCoordGT = (GT[0] + (GT[2] / 2), GT[1] + (GT[3] / 2))
    centerCoordT = (centerCoordT[0] / GT[2], centerCoordT[1] / GT[3])
    centerCoordGT = (centerCoordGT[0] / GT[2], centerCoordGT[1] / GT[3])
    return sqrt((centerCoordT[0] - centerCoordGT[0]) ** 2 + (centerCoordT[1] - centerCoordGT[1]) ** 2)


def CDist(T, GT):
    centerCoordT = (T[0] + (T[2] / 2), T[1] + (T[3] / 2))
    centerCoordGT = (GT[0] + (GT[2] / 2), GT[1] + (GT[3] / 2))
    return sqrt((centerCoordT[0] - centerCoordGT[0]) ** 2 + (centerCoordT[1] - centerCoordGT[1]) ** 2)


def IOU(T, GT):
    xA = max(T[0], GT[0])
    yA = max(T[1], GT[1])
    xB = min(T[0] + T[2], GT[0] + GT[2])
    yB = min(T[1] + T[3], GT[1] + GT[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    TArea = T[2] * T[3]
    GTArea = GT[2] * GT[3]
    return interArea / float(TArea + GTArea - interArea)


def F1(T, GT):
    xT = max(T[0], GT[0])
    yT = max(T[1], GT[1])
    xGT = min(T[2], GT[2])
    yGT = min(T[3], GT[3])
    interArea = max(0, xGT - xT + 1) * max(0, yGT - yT + 1)
    TArea = (T[2] - T[0] + 1) * (T[3] - T[1] + 1)
    GTArea = (GT[2] - GT[0] + 1) * (GT[3] - GT[1] + 1)
    p = interArea / TArea
    r = interArea / GTArea
    return 2 * p * r / (p + r)


def F(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def OTA(fn, fp, g):
    return 1 - (fn + fp) / g


def Analise(videos, vid_folder, tracker_t):
    d = {'iou': [0.0] * len(list(videos)), 'fps': [0.0] * len(list(videos)), 'dist': [0.0] * len(list(videos)),
         'normdist': [0.0] * len(list(videos)), }
    for j, seq_ID in enumerate(videos):
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
            fpS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

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
            cv2.putText(frame, "FPS : " + str(int(fpS)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            cv2.imshow("Tracking", frame)
            d.get("iou")[j] += IOU(bbox, anno_bbox)
            d.get("dist")[j] += CDist(bbox, anno_bbox)
            d.get("normdist")[j] += NCDist(bbox, anno_bbox)
            d.get("fps")[j] += fpS
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                sys.exit()
        d.get("iou")[j] /= len(frames_list)
        d.get("fps")[j] /= len(frames_list)
        d.get("dist")[j] /= len(frames_list)
        d.get("normdist")[j] /= len(frames_list)

        print(f" IOU = {d.get('iou')[j]}, FPS = {d.get('fps')[j]}, CDist = {d.get('dist')[j]},"
              f" NCDist = {d.get('normdist')[j]}")
        # cv2.imwrite(anno_file, frame)
    return d


tracker_type = "MOSSE"
chunk_folder = "F:\\Torents\\TRAIN_0".upper()
vid = tqdm(random.choices(os.listdir(os.path.join(chunk_folder, "frames")), k=10))
print(Analise(vid, chunk_folder, tracker_type))
# print(f"{tracker_type} tracker : Average IOU = {mean(iou)}, average FPS = {mean(fps)},",
#       f"average CDist = {mean(dst)}, average NCDist = {mean(ndst)}, score = {mean(iou) * mean(fps) / mean(dst)}")
