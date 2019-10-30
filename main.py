import os
import cv2
import numpy as np
from background import subtract_background_rolling_ball
# import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

base_dir = "data"
base_path = Path(base_dir)

dirs = os.listdir(base_path)
df = pd.DataFrame()

for dir in dirs:
    if not os.path.isdir(base_path / dir):
        continue
    files = os.listdir(base_path / dir)
    for f in files:
        if f[-4:] != ".tif":
            continue
        img = cv2.imread(str(base_path / dir / f), cv2.IMREAD_UNCHANGED)
        # plt.imshow(img.astype(np.float32))
        # plt.show()
        out, background = subtract_background_rolling_ball(img, 50, light_background=False,
                                                           use_paraboloid=False, do_presmooth=False)
        print(img.shape, img.dtype, np.average(img), np.average(out))
        if not (dir in df.columns):
            df[dir] = ""
        if not (f in df.index):
            df.loc[f] = np.nan

        df[dir][f] = np.average(out)

df.to_csv("output.csv")


