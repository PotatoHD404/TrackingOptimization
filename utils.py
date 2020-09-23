import cv2
import random
import string
import os
import sys
import json

import numpy as np
import pandas as pd

from math import sqrt
from tqdm import tqdm
from statistics import mean
from time import asctime, localtime


# from tkinter.filedialog import askopenfilename
def rndStr(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


def WriteToExcel(vid, results):
    writer = pd.ExcelWriter(f"{asctime(localtime()).replace(':', ' ')}.xlsx", engine='xlsxwriter')
    workbook = writer.book
    pd.DataFrame(data={}).to_excel(writer, sheet_name='Sheet1')
    worksheet = writer.sheets['Sheet1']

    worksheet.set_column('A:A', 20)
    worksheet.set_column('O:S', 20)
    # worksheet.set_row(3, 30)
    # worksheet.set_row(6, 30)
    # worksheet.set_row(7, 30)
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 2,
        'align': 'center',
        'valign': 'vcenter'})
    for i, result in enumerate(results):
        data = pd.DataFrame(data=result[2], index=vid)
        params = pd.DataFrame(data=result[1]["my_object"], index={0})
        if i == 0:
            data.index.name = "Video"
            params.to_excel(writer, sheet_name='Sheet1', startrow=1, startcol=13)
            data.to_excel(writer, sheet_name='Sheet1', startrow=1)
            worksheet.merge_range(f'A1:N1', result[0], merge_format)
        else:
            row = i * (len(vid) + 1) + 1
            params.to_excel(writer, sheet_name='Sheet1', startrow=row, startcol=13)
            data.to_excel(writer, sheet_name='Sheet1', startrow=row)
            worksheet.merge_range(f'A{row + 1}:N{row + 1}', result[0], merge_format)
    for col in ["B", "C", "D", "E", "F", "G", "H"]:
        worksheet.conditional_format(f'{col}3:{col}{len(results) * (len(vid) + 1) + 1}',
                                     {'type': '3_color_scale'})
        # Merge 3 cells.

    writer.save()


# pandas frame
# iv.dps = 10
def PrintV():
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    print(f"OpenCV v{major_ver}.{minor_ver}.{subminor_ver}")


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
    chunk_folder = "F:\\Torents\\TRAIN_0".upper()
    vid = random.choices(os.listdir(os.path.join(chunk_folder, "frames")), k=number)
    if progressBar:
        vid = tqdm(vid)
    return vid


def Analise(videos, result, ui=False, vid_folder="F:\\Torents\\TRAIN_0".upper()):
    data = {'ata': [0.0] * len(videos), 'F': [0.0] * len(videos), 'F1': [0.0] * len(videos), 'otp': [0.0] * len(videos),
            'ota': [0.0] * len(videos), 'deviation': [0.0] * len(videos), 'PBM': [0.0] * len(videos),
            'fps': [0.0] * len(videos), 'Ms': [0] * len(videos), 'fp': [0] * len(videos),
            'tp': [0] * len(videos), 'fn': [0] * len(videos), 'g': [0] * len(videos)}
    result[2] = data
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
        tracker_type = result[0]
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
        file1.writelines(json.dumps(result[1]))
        file1.close()
        tracker.read(cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ).getFirstTopLevelNode())
        os.remove(file_path)
        ok = tracker.init(frame, bbox)
        if not ok:
            print("Initialisation error")
            continue
        data["Ms"][j] += 1
        data["tp"][j] += 1
        data["ata"][j] += IOU(bbox, bbox)
        data["deviation"][j] += NCDist(bbox, bbox)
        data["F1"][j] += F1(bbox, bbox)
        data["PBM"][j] += 1 - L1(bbox, bbox)
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
                if ui:
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                # print("Tracking failure detected")
                if True:
                    data["fn"][j] += 1
                if ui:
                    cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)

            # Display tracker type on frame
            if ui:
                cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                            2)
                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fpS)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                            2)
                # Display result
                cv2.imshow("Tracking", frame)
            iou = IOU(bbox, anno_bbox)
            data["ata"][j] += iou
            data["deviation"][j] += NCDist(bbox, anno_bbox)
            data["fps"][j] += fpS
            data["F1"][j] += F1(bbox, anno_bbox)
            data["PBM"][j] += 1 - (L1(bbox, anno_bbox) if iou > 0 else Th(bbox, anno_bbox)) / Th(bbox, anno_bbox)
            if True:
                data["g"][j] += 1
            if iou > 0:
                data["Ms"][j] += 1
                data["tp"][j] += 1
            else:
                data["fp"][j] += 1

            # Exit if ESC pressed
            if ui:
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    sys.exit()
        data["F"][j] = F(data["tp"][j], data["fp"][j], data["fn"][j])
        data["F1"][j] /= len(frames_list)
        data["ota"][j] = 1 - (data["fp"][j] + data["fn"][j]) / data["g"][j]
        data["otp"][j] = data["ata"][j] / data["Ms"][j]
        data["ata"][j] /= len(frames_list)
        data["fps"][j] /= len(frames_list)
        data["deviation"][j] = 1 - data["deviation"][j] / data["Ms"][j]
        if data["deviation"][j] < 0:
            data["deviation"][j] = -1
        data["PBM"][j] /= len(frames_list)
        # print(f" IOU = {d['iou'][j] }, FPS = {d['fps'][j] }, CDist = {d['dist'][j] },"
        #       f" NCDist = {d['normdist'][j] }")
        # cv2.imwrite(anno_file, frame)
