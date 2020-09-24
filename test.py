from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm


def func(a):
    for i in range(1000):
        pbar.update()


pool = ThreadPool(8)
results = range(100)
pbar = tqdm(total=1000 * len(results))
pool.map_async(lambda p: func(p), results)
pool.close()
pool.join()
pbar.close()
