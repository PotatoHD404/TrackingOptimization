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
# Form, Window = uic.loadUiType("design.ui")
#
# app = QApplication(sys.argv)
# window = Window()
# form = Form()
# form.setupUi(window)
# window.show()
# app.exec_()

utils.PrintV()

vid = utils.GetDataSet(2)
results = [["MOSSE",
            {'my_object': {}}, {}],
           ["CSRT", {'my_object': {'padding': 3.0, 'template_size': 200.0, 'gsl_sigma': 1.0, 'hog_orientations': 9.0,
                                   'num_hog_channels_used': 18, 'hog_clip': 0.20000000298023224, 'use_hog': 1,
                                   'use_color_names': 1, 'use_gray': 1, 'use_rgb': 0, 'window_function': 'hann',
                                   'kaiser_alpha': 3.75, 'cheb_attenuation': 45.0, 'filter_lr': 0.019999999552965164,
                                   'admm_iterations': 4, 'number_of_scales': 33, 'scale_sigma_factor': 0.25,
                                   'scale_model_max_area': 512.0, 'scale_lr': 0.02500000037252903,
                                   'scale_step': 1.0199999809265137, 'use_channel_weights': 1,
                                   'weights_lr': 0.019999999552965164, 'use_segmentation': 1, 'histogram_bins': 16,
                                   'background_ratio': 2, 'histogram_lr': 0.03999999910593033,
                                   'psr_threshold': 0.03500000014901161}}, {}]]
utils.Analise(vid, results[0])
utils.Analise(vid, results[1])
# threads = []
# for tracker in results:
#     threads.append(threading.Thread(target=utils.Analise, args=(vid, tracker)))
# for j in threads:
#     j.start()
# for j in threads:
#     j.join()
utils.WriteToExcel(vid, results)
# print(f"{tracker_type} tracker : Average IOU = {mean(iou)}, average FPS = {mean(fps)},",
#       f"average CDist = {mean(dst)}, average NCDist = {mean(ndst)}, score = {mean(iou) * mean(fps) / mean(dst)}")
