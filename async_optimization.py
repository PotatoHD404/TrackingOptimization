import time
import random

from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours
import utils
from tqdm import tqdm
import multiprocessing
import threading
from multiprocessing.dummy import Pool as ThreadPool
from time import asctime, localtime
import asyncio
import threading

try:
    import json
    import tornado.ioloop
    import tornado.httpserver
    from tornado.web import RequestHandler
    import requests
except ImportError:
    raise ImportError(
        "In order to run this example you must have the libraries: " +
        "`tornado` and `requests` installed."
    )

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
        pool.apply_async(utils.Analise, args=[result])

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
    print(f"1,{winSize},{winSizeNCC},{res[0]}")
    return res[0]


class BayesianOptimizationHandler(RequestHandler):
    """Basic functionality for NLP handlers."""
    _bo = BayesianOptimization(
        f=black_box_function,
        pbounds={'winSize': (1, 100),
                 'winSizeNCC': (5, 100)}
    )
    _uf = UtilityFunction(kind="ei", kappa=10, xi=0.1)

    def post(self):
        """Deal with incoming requests."""
        body = tornado.escape.json_decode(self.request.body)

        try:
            self._bo.register(
                params=body["params"],
                target=body["target"],
            )
            # print("BO has registered: {} points.".format(len(self._bo.space)), end="\n\n")
        except KeyError:
            pass
        finally:
            suggested_params = self._bo.suggest(self._uf)

        self.write(json.dumps(suggested_params))


def run_optimization_app():
    asyncio.set_event_loop(asyncio.new_event_loop())
    handlers = [
        (r"/bayesian_optimization", BayesianOptimizationHandler),
    ]
    server = tornado.httpserver.HTTPServer(
        tornado.web.Application(handlers)
    )
    server.listen(9008)
    tornado.ioloop.IOLoop.instance().start()


def run_optimizer():
    global optimizers_config
    config = optimizers_config.pop()
    name = config["name"]
    colour = config["colour"]

    register_data = {}
    max_target = None
    for _ in range(500):
        status = name + " wants to register: {}.\n".format(register_data)

        resp = requests.post(
            url="http://localhost:9008/bayesian_optimization",
            json=register_data,
        ).json()
        target = black_box_function(**resp)

        register_data = {
            "params": resp,
            "target": target,
        }

        if max_target is None or target > max_target:
            max_target = target

        status += name + " got {} as target.\n".format(target)
        status += name + " will to register next: {}.\n".format(register_data)
        # print(colour(status), end="\n")

    global results
    results.append((name, max_target))
    # print(colour(name + " is done!"), end="\n\n")


if __name__ == "__main__":
    ioloop = tornado.ioloop.IOLoop.instance()
    optimizers_config = [
        {"name": "optimizer 1", "colour": Colours.red},
        {"name": "optimizer 2", "colour": Colours.green},
        {"name": "optimizer 3", "colour": Colours.blue},
        {"name": "optimizer 4", "colour": Colours.purple},
        {"name": "optimizer 5", "colour": Colours.black},
        {"name": "optimizer 6", "colour": Colours.red},
        {"name": "optimizer 7", "colour": Colours.cyan},
        {"name": "optimizer 8", "colour": Colours.yellow},
        {"name": "optimizer 9", "colour": Colours.yellow},
    ]

    app_thread = threading.Thread(target=run_optimization_app)
    app_thread.daemon = True
    app_thread.start()

    targets = (
        run_optimizer,
        run_optimizer,
        run_optimizer,
        run_optimizer,
        run_optimizer,
        run_optimizer,
        run_optimizer,
        run_optimizer,
        run_optimizer,
    )
    optimizer_threads = []
    for target in targets:
        optimizer_threads.append(threading.Thread(target=target))
        optimizer_threads[-1].daemon = True
        optimizer_threads[-1].start()

    results = []
    for optimizer_thread in optimizer_threads:
        optimizer_thread.join()

    for result in results:
        print(result[0], "found a maximum value of: {}".format(result[1]))

    ioloop.stop()
