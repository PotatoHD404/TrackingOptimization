import utils
import threading

utils.PrintV()

vid = utils.GetDataSet(1)
result = {"MOSSE": 0, "CSRT": 0}
threads = []
for tracker in result.keys():
    threads.append(threading.Thread(target=utils.Analise, args=(vid, tracker, result, True)))
for j in threads:
    j.start()
for j in threads:
    j.join()
utils.WriteToExcel(vid, result)
# print(f"{tracker_type} tracker : Average IOU = {mean(iou)}, average FPS = {mean(fps)},",
#       f"average CDist = {mean(dst)}, average NCDist = {mean(ndst)}, score = {mean(iou) * mean(fps) / mean(dst)}")
