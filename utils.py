import cv2
import random
import string
import os
import sys
import json
import multiprocessing
import numpy as np
import pandas as pd
import threading

from itertools import chain
from sklearn.model_selection import ParameterGrid
from math import sqrt
from tqdm import tqdm
from statistics import mean
from time import asctime, localtime
from multiprocessing.dummy import Pool as ThreadPool


def gen_grid(params):
    result = []
    for param in params:
        grids = ParameterGrid(param_grid=param)
        for grid in grids:
            result.append({'my_object': grid})
    return result


def col_str(n):
    res = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        res = chr(65 + remainder) + res
    return res


# from tkinter.filedialog import askopenfilename
def rndStr(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


def WriteToCSV(results):
    res = pd.DataFrame(data=results)
    filename = f"{asctime(localtime()).replace(':', ' ')}.csv"
    res.to_csv(filename, encoding='utf-8', index=False)


def WriteToExcel(vid, results):
    vid.append("avg")
    filename = f"{asctime(localtime()).replace(':', ' ')}.xlsx"
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
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
    maxCols = 0
    for i, result in enumerate(results):
        data = pd.DataFrame(data=result[2], index=vid)
        params = pd.DataFrame(data=result[1]["my_object"], index={0})
        dC = len(result[2]) + 1
        if maxCols < len(result[1]["my_object"]):
            maxCols = len(result[1]["my_object"])
        if i == 0:
            data.index.name = "Video"
            params.to_excel(writer, sheet_name='Sheet1', startrow=1, startcol=dC - 1)
            data.to_excel(writer, sheet_name='Sheet1', startrow=1)
            worksheet.merge_range(f'{col_str(1)}1:{col_str(dC)}1', result[0], merge_format)
        else:
            row = i * (len(vid) + 1) + 1
            params.to_excel(writer, sheet_name='Sheet1', startrow=row, startcol=dC - 1)
            data.to_excel(writer, sheet_name='Sheet1', startrow=row)
            worksheet.merge_range(f'{col_str(1)}{row + 1}:{col_str(dC)}{row + 1}', result[0], merge_format)
    worksheet.set_column(f'{col_str(dC)}:{col_str(dC + maxCols)}', 20)
    for col in chain(range(2, 10), range(dC + 1, dC + 1 + maxCols)):
        worksheet.conditional_format(f'{col_str(col)}3:{col_str(col)}{len(results) * (len(vid) + 1) + 1}',
                                     {'type': '3_color_scale'})
        # Merge 3 cells.

    writer.save()
    files.download(filename)


# pandas frame
# iv.dps = 10
def PrintV():
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    print(f"OpenCV v{major_ver}.{minor_ver}.{subminor_ver}\r\n")


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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def OTA(fn, fp, g):
    return 1 - (fn + fp) / g


def GetDataSet(number=10, progressBar=False):
    chunk_folder = r"F:\Torents\TRAIN_0"
    vid = random.choices(os.listdir(os.path.join(chunk_folder, "frames")), k=number)
    if progressBar:
        vid = tqdm(vid)
    return vid


def Analise(result, vid_folder=r"F:\Torents\TRAIN_0"):
    data = {'ATA': 0.0, 'F': 0.0, 'F1': 0.0, 'OTP': 0.0,
            'OTA': 0.0, 'Deviation': 0.0, 'PBM': 0.0,
            'FPS': 0.0, 'Ms': 0, 'fp': 0,
            'tp': 0, 'fn': 0, 'g': 0}
    tracker_type = result["my_object"]["tracker"]
    del result["my_object"]["tracker"]
    seq_ID = result["my_object"]["video"]
    del result["my_object"]["video"]
    frames_folder = os.path.join(vid_folder, "frames", seq_ID)
    anno_file = os.path.join(vid_folder, "anno", seq_ID + ".txt")
    anno = np.loadtxt(anno_file, delimiter=",")
    frames_list = [frame for frame in os.listdir(frames_folder) if
                   frame.endswith(".jpg")]
    if not len(anno) == len(frames_list):
        print("Not the same number of frames and annotation!")
        return
        # Define an initial bounding box
    bbox = (anno[0][0], anno[0][1], anno[0][2], anno[0][3])
    frame = cv2.imread(os.path.join(frames_folder, "0.jpg"))
    if tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    file_path = rndStr() + ".json"
    file1 = open(file_path, 'w')
    file1.writelines(json.dumps(result))
    file1.close()
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    tracker.read(fs.getFirstTopLevelNode())
    os.remove(file_path)
    ok = tracker.init(frame, bbox)
    if not ok:
        print("Initialisation error")
        return
    data["Ms"] += 1
    data["tp"] += 1
    data["ATA"] += IOU(bbox, bbox)
    data["Deviation"] += NCDist(bbox, bbox)
    data["F1"] += F1(bbox, bbox)
    data["PBM"] += 1 - L1(bbox, bbox)
    for i in range(1, len(frames_list)):
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
        else:
            # Tracking failure
            # print("Tracking failure detected")
            if True:
                data["fn"] += 1

        # Display tracker type on frame
        iou = IOU(bbox, anno_bbox)
        data["ATA"] += iou
        data["FPS"] += fpS
        data["F1"] += F1(bbox, anno_bbox)
        data["PBM"] += 1 - (L1(bbox, anno_bbox) if iou > 0 else Th(bbox, anno_bbox)) / Th(bbox, anno_bbox)
        if True:
            data["g"] += 1
        if iou >= 0.5:
            data["tp"] += 1
        elif iou > 0:
            data["OTP"] += iou
            data["Deviation"] += NCDist(bbox, anno_bbox)
            data["Ms"] += 1
        else:
            data["fp"] += 1

        # Exit if ESC pressed
        # print(tracker_type, i)
    data["F"] = F(data["tp"], data["fp"], data["fn"])
    data["F1"] /= len(frames_list)
    data["OTA"] = 1 - (data["fp"] + data["fn"]) / data["g"]
    data["OTP"] /= data["Ms"]
    data["ATA"] /= len(frames_list)
    data["FPS"] /= len(frames_list)
    data["Deviation"] = 1 - data["Deviation"] / data["Ms"]
    if data["Deviation"] < -1:
        data["Deviation"] = -1
    data["PBM"] /= len(frames_list)
    temp = result["my_object"]
    del result["my_object"]
    result.update({"video": seq_ID})
    result.update(data)
    result.update(temp)
