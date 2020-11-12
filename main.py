import utils
from tqdm import tqdm
import multiprocessing
import threading
from multiprocessing.dummy import Pool as ThreadPool
from time import asctime, localtime


def update1(*a):
    pbar.update()


utils.PrintV()

# vid = [
#     "GceLsS4AwH8_1",
#     "j93wwDC_a2I_0",
#     "BpXhq5Awd3U_0",
#     "EwlCKB77dYo_3",
#     "JR0QfXOOmaA_0",
#     "n0tx4V2rF3I_1",
#     "EwlCKB77dYo_4",
#     "ZlEiOICCDdc_0",
#     "f8rXEKktSCg_0",
#     "afU2vHgUvaw_2",
#     "ckFwzL1Ot94_0",
#     "C3lwMd_rlG0_0",
#     "afU2vHgUvaw_3",
#     "hBHt6mnfUeo_0",
#     "h-pm7wD31Ss_3",
#     "CWCfCeYh2bA_1",
#     "TalhQQ9B7vc_0",
#     "cH9u1pCWp2U_0",
#     "afU2vHgUvaw_7",
#     "aCNvyXSuG6w_0",
#     "gOWc7VBEwMo_0",
#     "EmvEUer4CVc_0",
#     "h-pm7wD31Ss_0",
#     "EwlCKB77dYo_2"
# ]
vid = [
    "TalhQQ9B7vc_0"
]
grid = [{"pointsInGrid": [10],
         "winSize": [[3, 3]],
         "maxLevel": [5],
         "termCriteria_maxCount": [20, 10, 5],
         "termCriteria_epsilon": [2.9999999999999999e-01],
         "winSizeNCC": [[30, 30], [10, 10], [100, 100], [1000, 1000], [1.0, 1.0], [2.0, 2.0], [0.5, 0.5],
                        [500.0, 500.0], [250.0, 250.0], [400.0, 400.0]],
         "maxMedianLengthOfDisplacementDifference": [10.0],
         'video': vid,
         'tracker': ["MEDIANFLOW"]}]
# grid = [{'padding': [2.0],
#                   'template_size': [200.0],
#                   'gsl_sigma': [1.0],
#                   'hog_orientations': [8.0],
#                   'num_hog_channels_used': [18],
#                   'hog_clip': [0.2],
#                   'use_hog': [1],
#                   'use_color_names': [1],
#                   'use_gray': [1],
#                   'use_rgb': [0],
#                   'window_function': ['hann'],
#                   'kaiser_alpha': [3.75],
#                   'cheb_attenuation': [45.0],
#                   'filter_lr': [0.02],
#                   'admm_iterations': [4],
#                   'number_of_scales': [33],
#                   'scale_sigma_factor': [0.25],
#                   'scale_model_max_area': [512.0],
#                   'scale_lr': [0.025],
#                   'scale_step': [1.02],
#                   'use_channel_weights': [1],
#                   'weights_lr': [0.02],
#                   'use_segmentation': [1],
#                   'histogram_bins': [16],
#                   'background_ratio': [2],
#                   'histogram_lr': [0.04],
#                   'psr_threshold': [0.035],
#                   'video': ["EwlCKB77dYo_4"],
#                   'tracker': ["CSRT"]}]
# threads = []
# for result in results:
#   threads.apply(threading.Thread(target=Analise, args=[vid, result])
# for j in threads:
#     j.start()
# for j in threads:
#     j.join()

results = utils.gen_grid(grid)
pool = ThreadPool()

# print(results[1])
# Analise(results[1])
# print(results[1])
pbar = tqdm(total=len(results))
for result in results:
    pool.apply_async(utils.Analise, args=[result], callback=update1)

# pool.map(lambda p: Analise(vid, p), results)
pool.close()
pool.join()
pbar.close()
# p.close()
# p.join()
utils.WriteToCSV(results)
print("done")
