import utils
from tqdm import tqdm
import multiprocessing
import threading
from multiprocessing.dummy import Pool as ThreadPool
from time import asctime, localtime
from bayes_opt import BayesianOptimization
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
import math
import numpy as np


def update1(*a):
    ""


# utils.WriteToCSV(results, "Last")
# pbar.update()


# pip install opencv-contrib-python==4.1.0.25
utils.PrintV()

vid = [
    # "GceLsS4AwH8_1",
    # "j93wwDC_a2I_0",
    "BpXhq5Awd3U_0",
    "EwlCKB77dYo_3",
    "JR0QfXOOmaA_0",
    # "n0tx4V2rF3I_1",
    "EwlCKB77dYo_4",
    # "ZlEiOICCDdc_0",
    # "f8rXEKktSCg_0",
    "afU2vHgUvaw_2",
    "ckFwzL1Ot94_0",
    "C3lwMd_rlG0_0",
    "afU2vHgUvaw_3",
    "hBHt6mnfUeo_0",
    "h-pm7wD31Ss_3",
    "CWCfCeYh2bA_1",
    "TalhQQ9B7vc_0",
    "cH9u1pCWp2U_0",
    "afU2vHgUvaw_7",
    "aCNvyXSuG6w_0",
    "gOWc7VBEwMo_0",
    "EmvEUer4CVc_0",
    "h-pm7wD31Ss_0",
    "EwlCKB77dYo_2"
]
vid = vid[:2]


# vid = [
#     "TalhQQ9B7vc_0"
# ]


def black_box_function(winSize,
                       winSizeNCC, target="ATA"):
    # optimizer.probe(
    #     params={"pointsInGrid": 10,
    #             "winSize": 3,
    #             "maxLevel": 5,
    #             "termCriteria_maxCount": 20,
    #             "termCriteria_epsilon": 2.9999999999999999e-01,
    #             "winSizeNCC": 30,
    #             "maxMedianLengthOfDisplacementDifference": 10.0},
    #     lazy=True,
    # )
    grid = [{'maxLevel': [5],
             'maxMedianLengthOfDisplacementDifference': [10.0],
             'pointsInGrid': [10],
             'termCriteria_epsilon': [2.9999999999999999e-01], 'termCriteria_maxCount': [20],
             'tracker': ['MEDIANFLOW'],
             'video': vid, 'winSize': [[winSize, winSize]], 'winSizeNCC': [[winSizeNCC, winSizeNCC]]}]
    results = utils.gen_grid(grid)
    pool = ThreadPool(8)

    # print(results[1])
    # Analise(results[1])
    # print(results[1])
    # pbar = tqdm(total=len(results))
    res = [0.0, 0]
    for result in results:
        pool.apply_async(utils.Analise, args=[result], callback=update1)

    # pool.map(lambda p: Analise(vid, p), results)
    pool.close()
    pool.join()
    # [print(i["ATA"]) for i in results]
    for result in results:
        try:
            res[0] += result[target]
            res[1] += 1
        except:
            return -1
    res[0] /= res[1]
    return res[0]


optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={'winSize': (1, 100),
             'winSizeNCC': (5, 100)},
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
)

optimizer.maximize(
    n_iter=1000, acq="poi", xi=1e-1
)
# bounds = {'winSize': ('cont', [1, 100]),
#           'winSizeNCC': ('cont', [5, 100])}
# cov = matern32()
# gp = GaussianProcess(cov)
# acq = Acquisition(mode='ExpectedImprovement')
#
# # np.random.seed(1337)
# gpgo = GPGO(gp, acq, black_box_function, bounds)
# gpgo.run(max_iter=1000)
# print(optimizer.max)
# optimizer.maximize(
#     n_iter=500, acq="poi", xi=1e-1
# )
# print(optimizer.max)
# optimizer.maximize(n_iter=500, acq="ucb", kappa=0.1)
# print(optimizer.max)
# # grid = [{'padding': [2.0],
# #                   'template_size': [200.0],
# #                   'gsl_sigma': [1.0],
# #                   'hog_orientations': [8.0],
# #                   'num_hog_channels_used': [18],
# #                   'hog_clip': [0.2],
# #                   'use_hog': [1],
# #                   'use_color_names': [1],
# #                   'use_gray': [1],
# #                   'use_rgb': [0],
# #                   'window_function': ['hann'],
# #                   'kaiser_alpha': [3.75],
# #                   'cheb_attenuation': [45.0],
# #                   'filter_lr': [0.02],
# #                   'admm_iterations': [4],
# #                   'number_of_scales': [33],
# #                   'scale_sigma_factor': [0.25],
# #                   'scale_model_max_area': [512.0],
# #                   'scale_lr': [0.025],
# #                   'scale_step': [1.02],
# #                   'use_channel_weights': [1],
# #                   'weights_lr': [0.02],
# #                   'use_segmentation': [1],
# #                   'histogram_bins': [16],
# #                   'background_ratio': [2],
# #                   'histogram_lr': [0.04],
# #                   'psr_threshold': [0.035],
# #                   'video': ["EwlCKB77dYo_4"],
# #                   'tracker': ["CSRT"]}]
# # threads = []
# # for result in results:
# #   threads.apply(threading.Thread(target=Analise, args=[vid, result])
# # for j in threads:
# #     j.start()
# # for j in threads:
# #     j.join()
#
# results = utils.gen_grid(grid)
# print(results)
# pool = ThreadPool(8)
#
# # print(results[1])
# # Analise(results[1])
# # print(results[1])
# pbar = tqdm(total=len(results))
# for result in results:
#     pool.apply_async(utils.Analise, args=[result], callback=update1)
# grid = [{'maxLevel': [5],
#          'maxMedianLengthOfDisplacementDifference': [10.0],
#          'pointsInGrid': [10],
#          'termCriteria_epsilon': [2.9999999999999999e-01], 'termCriteria_maxCount': [20],
#          'tracker': ['MEDIANFLOW'],
#          'video': [vid], 'winSize': [[3, 3]], 'winSizeNCC': [[500, 500]]}]
# results = utils.gen_grid(grid)
# pool = ThreadPool(8)

# print(results[1])
# Analise(results[1])
# print(results[1])
# pbar = tqdm(total=len(results))
# res = [0.0, 0]
# for result in results:
#     pool.apply_async(utils.Analise, args=[result], callback=update1)
# # pool.map(lambda p: Analise(vid, p), results)
# pool.close()
# pool.join()
# # pbar.close()
# print([i["FPS"] for i in results])
# p.close()
# p.join()
# utils.WriteToCSV(results)
print("done")
