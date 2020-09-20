import cv2
import sys
import os
from tqdm import tqdm
import random
from time import sleep
import numpy as np
from tkinter.filedialog import askopenfilename

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
print(major_ver, minor_ver, subminor_ver)

# Set up tracker.
# Instead of MIL, you can also use
print("cv2")
tracker_type = "CSRT"
trackers = dict(TLD=cv2.TrackerTLD_create(),
                MOSSE=cv2.TrackerMOSSE_create(),
                GOTURN=cv2.TrackerGOTURN_create(),
                BOOSTING=cv2.TrackerBoosting_create(),
                MIL=cv2.TrackerMIL_create(),
                KCF=cv2.TrackerKCF_create(),
                MEDIANFLOW=cv2.TrackerMedianFlow_create(),
                CSRT=cv2.TrackerCSRT_create())

tracker = trackers.get(tracker_type)

# for key in trackers.keys():
#     trackers.get(key).save(f'settings_{key}.json')
# Read video
# F:\Torents\TRAIN_0
chunk_folder = "F:\\Torents\\TRAIN_0".upper()
list_sqruences = random.choices(os.listdir(os.path.join(chunk_folder, "frames")), k=100)
for seq_ID in tqdm(list_sqruences):
    frames_folder = os.path.join(chunk_folder, "frames", seq_ID)
    anno_file = os.path.join(chunk_folder, "anno", seq_ID + ".txt")
    anno = np.loadtxt(anno_file, delimiter=",")
    frames_list = [frame for frame in os.listdir(frames_folder) if
                   frame.endswith(".jpg")]
    if not len(ArrayBB) == len(frames_list):
        print("Not the same number of frames and annotation!")
        continue
    for i in range(len(frames_list)):
        frame_file = str(i) + ".jpg"
        imgs_file = os.path.join(frames_folder, frame_file)
        imbb_file = os.path.join(frames_BB_folder, frame_file)

        frame = cv2.imread(imgs_file)
        height, width, channels = frame.shape

        x_min = int(ArrayBB[i][0])
        x_max = int(ArrayBB[i][2] + ArrayBB[i][0])
        y_min = int(ArrayBB[i][1])
        y_max = int(ArrayBB[i][3] + ArrayBB[i][1])

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.imwrite(imbb_file, frame)

video = cv2.VideoCapture(askopenfilename(filetypes=[("Video files", ".mp4 .avi .mov .webm")]))

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

# Define an initial bounding box
# bbox = (287, 23, 86, 320)

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)
# fs = cv2.FileStorage("settings.yaml", cv2.FILE_STORAGE_WRITE)
# tracker.write(fs)
# fs.release()


while True:
    # Read a new frame

    ok, frame = video.read()
    if not ok:
        break

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
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
