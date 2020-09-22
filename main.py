import utils
import threading
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication
import cv2
# tracker = cv2.TrackerCSRT_create()
# file_name = 'tracker_params.json'
#
# tracker.save(file_name)
Form, Window = uic.loadUiType("design.ui")

app = QApplication(sys.argv)
window = Window()
form = Form()
form.setupUi(window)
window.show()
app.exec_()

utils.PrintV()

vid = utils.GetDataSet(2)
result = {"MOSSE": 0, "CSRT": 0}
threads = []
for tracker in result.keys():
    threads.append(threading.Thread(target=utils.Analise, args=(vid, tracker, result)))
for j in threads:
    j.start()
for j in threads:
    j.join()
utils.WriteToExcel(vid, result)
# print(f"{tracker_type} tracker : Average IOU = {mean(iou)}, average FPS = {mean(fps)},",
#       f"average CDist = {mean(dst)}, average NCDist = {mean(ndst)}, score = {mean(iou) * mean(fps) / mean(dst)}")
