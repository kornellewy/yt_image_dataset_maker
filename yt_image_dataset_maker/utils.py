import os
from pathlib import Path
import csv
from typing import List
import re
import pandas as pd
import numpy as np

import cv2


def variance_of_laplacian(image: np.ndarray) -> int:
    # https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    return cv2.Laplacian(image, cv2.CV_64F).var()


def yield_files_with_extensions(folder_path: str, file_extension: str) -> str:
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                yield os.path.join(root, file)


def load_images2(path: str) -> list:
    images_exts = (".jpg", ".jpeg", ".png")
    return list(yield_files_with_extensions(path, images_exts))


def save_dict_to_csv(csv_path: str, list_of_dict_to_log: List[dict]) -> None:
    csv_path = Path(csv_path)
    file_exist = False
    if csv_path.exists():
        file_exist = True
    with open(csv_path, "+a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list_of_dict_to_log[0].keys())
        if not file_exist:
            writer.writeheader()
        for data_dict in list_of_dict_to_log:
            data_dict = clean_data_dict(data_dict)
            writer.writerow(data_dict)


def clean_data_dict(data_dict: dict) -> dict:
    new_data_dict = {}
    for key, value in data_dict.items():
        if key == "url":
            new_data_dict.update({key: value})
        else:
            new_data_dict.update({key: clean_string(value)})
    return new_data_dict


def clean_string(string: str) -> str:
    string = string.lower()
    print(string)
    # string = " ".join(re.match("(\w+\s\w+)", string).groups())
    print(string)
    encoded_string = string.encode("ascii", "ignore")
    string = encoded_string.decode()
    string = string.replace(" ", "_")
    return string


def save_dict_to_csv2(csv_path: str, list_of_dict_to_log: List[dict]) -> None:
    csv_path = Path(csv_path)
    df = pd.DataFrame(list_of_dict_to_log)
    df.to_csv(csv_path, index=False, header=not csv_path.exists(), mode="a")


# read file with yt urls
def read_dict_from_csv(csv_path: str) -> List[dict]:
    df = pd.read_csv(csv_path)
    return df.to_dict("records")
