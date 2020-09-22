import cv2
import sys
import os
import random

import numpy as np
import pandas as pd

from statistics import mean
from math import sqrt
from time import *
from tqdm import tqdm
from datetime import date
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


def Th(T, GT):
    return (T[2] + T[3] + GT[2] + GT[3]) / 2


def L1(T, GT):
    centerCoordT = (T[0] + (T[2] / 2), T[1] + (T[3] / 2))
    centerCoordGT = (GT[0] + (GT[2] / 2), GT[1] + (GT[3] / 2))
    return abs(centerCoordT[0] - centerCoordGT[0] + centerCoordT[1] - centerCoordGT[1])


def IOU(T, GT):
    xT = max(T[0], GT[0])
    yT = max(T[1], GT[1])
    xGT = min(T[0] + T[2], GT[0] + GT[2])
    yGT = min(T[1] + T[3], GT[1] + GT[3])
    interArea = max(0, xGT - xT) * max(0, yGT - yT)
    TArea = T[2] * T[3]
    GTArea = GT[2] * GT[3]
    return interArea / float(TArea + GTArea - interArea)


def F1(T, GT):
    xT = max(T[0], GT[0])
    yT = max(T[1], GT[1])
    xGT = min(T[0] + T[2], GT[0] + GT[2])
    yGT = min(T[1] + T[3], GT[1] + GT[3])
    interArea = max(0, xGT - xT) * max(0, yGT - yT)
    TArea = T[2] * T[3]
    GTArea = GT[2] * GT[3]
    if not interArea > 0:
        return 0
    p = interArea / TArea
    r = interArea / GTArea
    return 2 * p * r / (p + r)


def F(tp, fp, fn):
    if tp == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def OTA(fn, fp, g):
    return 1 - (fn + fp) / g


def GetDataSet(number=10, progressBar=False):
    chunk_folder = "F:\\Torents\\TRAIN_0".upper()
    vid = random.choices(os.listdir(os.path.join(chunk_folder, "frames")), k=number)
    if progressBar:
        vid = tqdm(vid)
    return vid


def Analise(videos, tracker_t, res, ui=False, vid_folder="F:\\Torents\\TRAIN_0".upper()):
    d = {'ata': [0.0] * len(videos), 'F': [0.0] * len(videos), 'F1': [0.0] * len(videos), 'otp': [0.0] * len(videos),
         'ota': [0.0] * len(videos), 'deviation': [0.0] * len(videos), 'PBM': [0.0] * len(videos),
         'fps': [0.0] * len(videos), 'Ms': [0] * len(videos), 'fp': [0] * len(videos),
         'tp': [0] * len(videos), 'fn': [0] * len(videos), 'g': [0] * len(videos)}
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
                if True:
                    d["fn"][j] += 1
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                            2)

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fpS)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            cv2.imshow("Tracking", frame)
            iou = IOU(bbox, anno_bbox)
            d["ata"][j] += iou
            d["deviation"][j] += NCDist(bbox, anno_bbox)
            d["fps"][j] += fpS
            d["F1"][j] += F1(bbox, anno_bbox)
            d["PBM"][j] += 1 - (L1(bbox, anno_bbox) if iou > 0 else Th(bbox, anno_bbox)) / Th(bbox, anno_bbox)
            if True:
                d["g"][j] += 1
            if iou > 0:
                d["Ms"][j] += 1
                d["tp"][j] += 1
            else:
                d["fp"][j] += 1

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                sys.exit()
        d["F"][j] = F(d["tp"][j], d["fp"][j], d["fn"][j])
        d["F1"][j] /= len(frames_list)
        d["ota"][j] = 1 - (d["fp"][j] + d["fn"][j]) / d["g"][j]
        d["otp"][j] = d["ata"][j] / (d["Ms"][j] + 0.0001)
        d["ata"][j] /= len(frames_list)
        d["fps"][j] /= len(frames_list)
        d["deviation"][j] = 1 - d["deviation"][j] / (d["Ms"][j] + 0.0001)
        d["PBM"][j] /= len(frames_list)
        # print(f" IOU = {d['iou'][j] }, FPS = {d['fps'][j] }, CDist = {d['dist'][j] },"
        #       f" NCDist = {d['normdist'][j] }")
        # cv2.imwrite(anno_file, frame)
        res[tracker_t] = d


vid = GetDataSet()
result = {"MOSSE": 0}
threads = []
for tracker in result.keys():
    threads.append(threading.Thread(target=Analise, args=(vid, tracker, result)))
for j in threads:
    j.start()
for j in threads:
    j.join()

df = pd.DataFrame(data=result, index=vid)
df.index.name = "Video"
print(df)
writer = pd.ExcelWriter(f"{asctime(localtime()).replace(':', ' ')}.xlsx", engine='xlsxwriter')
workbook = writer.book
pd.DataFrame(data={}).to_excel(writer, sheet_name='Sheet1')
worksheet = writer.sheets['Sheet1']

worksheet.set_column('A:A', 20)
# worksheet.set_row(3, 30)
# worksheet.set_row(6, 30)
# worksheet.set_row(7, 30)
merge_format = workbook.add_format({
    'bold': 1,
    'border': 2,
    'align': 'center',
    'valign': 'vcenter'})
for i in range(10):
    if i == 0:
        df.to_excel(writer, sheet_name='Sheet1', startrow=1)
        worksheet.merge_range(f'A1:N1', tracker_type, merge_format)
    else:
        row = i * (len(vid) + 1) + 1
        df.to_excel(writer, sheet_name='Sheet1', startrow=row)
        worksheet.merge_range(f'A{row + 1}:N{row + 1}', tracker_type, merge_format)
    # Merge 3 cells.

writer.save()
# print(f"{tracker_type} tracker : Average IOU = {mean(iou)}, average FPS = {mean(fps)},",
#       f"average CDist = {mean(dst)}, average NCDist = {mean(ndst)}, score = {mean(iou) * mean(fps) / mean(dst)}")
