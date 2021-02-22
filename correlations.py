import pandas as pd

df = pd.read_csv(
    r"C:\Users\korna\Documents\Учёба\Проектная практика\TrackingOptimization\Thu Nov 12 16 54 00 2020.csv")
df = df.drop(
    columns=["video", "ATA", "F", "F1", "OTP", "OTA", "Deviation", "PBM", "Ms", "fp", "tp", "fn", "g", "maxLevel",
             "maxMedianLengthOfDisplacementDifference", "pointsInGrid", "termCriteria_epsilon", "termCriteria_maxCount",
             "winSize"])
print(df.interpolate())
