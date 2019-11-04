import os
import cv2
import numpy as np
from background import subtract_background_rolling_ball
# import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def analyze_images(folder, cache_file=None):
    base_path = Path(folder)

    dirs = os.listdir(base_path)
    df = pd.DataFrame()

    for dir in dirs:
        if not os.path.isdir(base_path / dir):
            continue
        files = os.listdir(base_path / dir)
        for f in files[:500]:
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
    if cache_file is not None:
        df.to_csv(cache_file)
    return df


def analyze_images_group(folder, cache_file=None):
    base_path = Path(folder)

    dirs = os.listdir(base_path)
    df = pd.DataFrame()

    for i, dir in enumerate(dirs):
        if not os.path.isdir(base_path / dir):
            continue
        files = os.listdir(base_path / dir)
        image = None
        image_num = 0
        for f in files[:]:
            if f[-4:] != ".tif":
                continue
            img = cv2.imread(str(base_path / dir / f), cv2.IMREAD_UNCHANGED)
            if image is None:
                image = img.astype(np.int32)
            else:
                image += img
            image_num += 1
        _, background = subtract_background_rolling_ball(image, 50, light_background=False,
                                                         use_paraboloid=False, do_presmooth=False)
        background = background // image_num
        for f in files[:]:
            if f[-4:] != ".tif":
                continue
            img = cv2.imread(str(base_path / dir / f), cv2.IMREAD_UNCHANGED)
            out = np.clip(img - background, 0, 1000000)
            # print(img.shape, img.dtype, np.average(img), np.average(out))
            if not (dir in df.columns):
                df[dir] = ""
            if not (f in df.index):
                df.loc[f] = np.nan

            df[dir][f] = np.average(out)
        print("Finished folder " + str(i) + " of " + str(len(dirs)))
    if cache_file is not None:
        df.to_csv(cache_file)
    return df


def normalize_data(input, cache_file=None):
    if isinstance(input, Path) or isinstance(input, str):
        df = pd.read_csv(input, index_col=0)
    else:
        df = input

    columns_rename = {x: x.split(" ")[1] for x in df.columns}
    df.rename(columns=columns_rename, inplace=True)

    columns_rename = {x: (x[1:] + x[0]) for x in df.columns}
    df.rename(columns=columns_rename, inplace=True)

    df = df.reindex(sorted(df.columns), axis=1)

    columns_rename = {v: k for k, v in columns_rename.items()}
    df.rename(columns=columns_rename, inplace=True)

    for col in df.columns:
        if 'C' in col:
            substract = df[col].iloc[0]
        else:
            substract = df[col].iloc[5]
        df[col] -= substract

    max_Cs = {}

    for col in df.columns:
        if 'C' in col:
            max_Cs[col[1:]] = df[col].max()

    for col in df.columns:
        df[col] /= max_Cs[col[1:]]

    print(max_Cs)

    if cache_file is not None:
        df.to_csv(cache_file)
    return df


def assemble_results(input, result_file):
    if isinstance(input, Path) or isinstance(input, str):
        df = pd.read_csv(input, index_col=0)
    else:
        df = input

    df_final = pd.DataFrame()

    for col in df.columns:
        i = col[0]
        c = col[1:]
        if not (c in df_final.columns):
            df_final[c] = ""
        if not (i in df_final.index):
            df_final.loc[i] = np.nan
        df_final[c][i] = df[col].max()

    if result_file is not None:
        df_final.to_csv(result_file)


base_dir = "G:\\My Drive\\PhD\\02_In vitro screening\\Kaitleen cmps\\Ca2+\\191024_AS003a\\2019-10-24_000"
analyze_images_group(base_dir, "output1.csv")
normalize_data("output1.csv", "output2.csv")
assemble_results("output2.csv", "final_output.csv")
