import pandas as pd

print(dict(pd.read_csv(
    r"C:\Users\korna\Documents\Учёба\Проектная практика\TrackingOptimization\bruhh.csv").corr(
    'pearson')))
