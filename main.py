import utils
import sys

from multiprocessing.dummy import Pool as ThreadPool
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication
from tqdm import tqdm


# Form, Window = uic.loadUiType("design.ui")
#
# app = QApplication(sys.argv)
# window = Window()
# form = Form()
# form.setupUi(window)
# window.show()
# app.exec_()
def update(*a):
    pbar.update()


utils.PrintV()

vid = utils.GetDataSet(5)
grid = [("MOSSE", {}), ("CSRT", {'padding': [2.0, 1.0],
                                 'template_size': [200.0, 300.0],
                                 'gsl_sigma': [1.0, 2.0],
                                 'hog_orientations': [8.0, 9.0],
                                 'num_hog_channels_used': [18],
                                 'hog_clip': [0.2],
                                 'use_hog': [1],
                                 'use_color_names': [1],
                                 'use_gray': [1],
                                 'use_rgb': [0],
                                 'window_function': ['hann'],
                                 'kaiser_alpha': [3.75],
                                 'cheb_attenuation': [45.0],
                                 'filter_lr': [0.02],
                                 'admm_iterations': [4],
                                 'number_of_scales': [33],
                                 'scale_sigma_factor': [0.25],
                                 'scale_model_max_area': [512.0],
                                 'scale_lr': [0.025],
                                 'scale_step': [1.02],
                                 'use_channel_weights': [1],
                                 'weights_lr': [0.02],
                                 'use_segmentation': [1],
                                 'histogram_bins': [16],
                                 'background_ratio': [2],
                                 'histogram_lr': [0.04],
                                 'psr_threshold': [0.035]})]
results = utils.gen_grid(grid)
threads = []
pool = ThreadPool(8)
pbar = tqdm(total=len(results))
for result in results:
    pool.apply_async(utils.Analise, args=(vid, result), callback=update)
# pool.map(lambda p: utils.Analise(vid, p), results)

pool.close()
pool.join()
pbar.close()
utils.WriteToExcel(vid, results)
print("done")
