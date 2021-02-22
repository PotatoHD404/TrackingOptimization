from bayes_opt import BayesianOptimization
import csv
from math import exp
from decimal import *
import math
import numpy as np
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO


def black_box_function(params):
    params = list(params.values())
    # print(params)
    res = 0
    with open('baesion1.txt', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for data in reader:
            data = list(data.values())
            tmp = Decimal(0.0)
            tmp1 = Decimal(data[1])
            for i, j in enumerate(data[2:]):
                tmp += exponents(params[i], Decimal(j))
            res += 1 - abs(tmp1 - tmp / tmp1)
    if res < 0 or not math.isfinite(res) or math.isnan(res) or math.isinf(res) or (res != res):
        print(res)
        return -res.as_tuple()[2]
    return res


def exponents(a, x):
    a = list(map(Decimal, a))
    return a[0] * (-a[1] * x).exp() - a[2] * (-a[3] * x).exp() + a[4] * (-a[5] * x).exp()


bounds = {}
params = ""
for i in range(1, 8):
    for j in range(1, 7):
        bounds[f"p{i}_{j}"] = ('cont', [-1000, 1000])
        params += f"p{i}_{j},"
params = params[:-1]
arrayer = '   arr = {}\n'
for i in params.split(','):
    tmp = i.split('_')
    if not arrayer.__contains__(tmp[0]):
        arrayer += f"   arr['{tmp[0]}'] = []\n"
    arrayer += f"   arr['{tmp[0]}'].append({i})\n"

func = f"""def bbf({params}):
{arrayer}
   return black_box_function(arr)
"""

# print(func)
exec(func)
cov = matern32()
gp = GaussianProcess(cov)
acq = Acquisition(mode='ExpectedImprovement')

np.random.seed(1337)
gpgo = GPGO(gp, acq, black_box_function, bounds)
gpgo.run(max_iter=1000)
# optimizer = BayesianOptimization(
#     f=bbf,
#     pbounds=bounds,
#     verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
# )
#
# # optimizer.probe(
# #     params={"pointsInGrid": 10,
# #             "winSize": 3,
# #             "maxLevel": 5,
# #             "termCriteria_maxCount": 20,
# #             "termCriteria_epsilon": 2.9999999999999999e-01,
# #             "winSizeNCC": 30,
# #             "maxMedianLengthOfDisplacementDifference": 10.0},
# #     lazy=True,
# # )
# optimizer.maximize(
#     n_iter=1500
# )
# print(optimizer.max)
