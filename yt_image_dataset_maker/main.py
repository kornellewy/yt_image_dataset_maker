import os
import shutil
from pathlib import Path
from pprint import pprint
import csv
import time
from PIL import Image
from datetime import datetime
import re
import numpy as np
from typing import List

import cv2
import torch
from pytube import YouTube
from facenet_pytorch import MTCNN

from utils import (
    variance_of_laplacian,
    load_images2,
    clean_string,
    read_dict_from_csv,
    save_dict_to_csv2,
)


# cut is per frame
def video_to_frames(
    input_loc: str, output_loc: str, number_of_images_to_log=1000
) -> List[str]:
    """
    https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    #
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # calculate per hom much frames we willo save
    log_interval = video_length // number_of_images_to_log
    if log_interval < 1:
        log_interval = 1
    print("log_interval ", log_interval)
    # Start converting the video
    images_paths = []
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        if count % log_interval == 0:
            # Write the results back to output location.
            image_path = output_loc + "/%#05d.jpg" % (count + 1)
            cv2.imwrite(image_path, frame)
            images_paths.append(image_path)
        count = count + 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds forconversion." % (time_end - time_start))
            break
    return images_paths


# remove blure images
def remove_blure_images(dir_path):
    images_paths = load_images2(dir_path)
    for image_path in images_paths:
        image = cv2.imread(image_path)
        if check_blure_image(image):
            print(image_path)


def check_blure_image(image: np.ndarray) -> bool:
    blure = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm < 1000:
        blure = True
    return blure


# find all faces in images
def find_faces(
    dir_path: str,
    face_dir_path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[str]:
    mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.8, 0.85, 0.85])
    images_paths = load_images2(dir_path)
    face_dir_path = Path(face_dir_path)
    face_dir_path.mkdir(exist_ok=True)
    faces_images_paths = []
    for image_path in images_paths:
        image_name = Path(image_path).stem
        image = Image.open(image_path)
        bboxes, _ = mtcnn.detect(image)
        if isinstance(bboxes, np.ndarray):
            for bbox_idx, bbox in enumerate(bboxes):
                face_bbox = image.crop(bbox)
                bbox_str = ",".join(["{:.2f}".format(x) for x in bbox])
                face_bbox_path = face_dir_path.joinpath(
                    f"image_name_{image_name}_bbox_idx_{bbox_idx}_bboxcord_{bbox_str}.jpg"
                )
                face_bbox.save(face_bbox_path)
                faces_images_paths.append(face_bbox_path)
                # brak bd we want only bigest face on image
                break
    return faces_images_paths


def download_yt_video(yt_video_url: str, yt_video_save_dir: str) -> str:
    yt_video = YouTube(yt_video_url)
    max_resolution_tag = 0
    max_resolution = 0
    yt_video_filename = clean_and_define_video_name(
        yt_video.streams[0].default_filename
    )
    for stream in yt_video.streams:
        if stream.resolution:
            resolution = int(stream.resolution[:-1])
            tag = stream.itag
            if max_resolution < resolution:
                max_resolution = resolution
                max_resolution_tag = tag
    yt_video.streams.get_by_itag(max_resolution_tag).download(
        yt_video_save_dir, yt_video_filename
    )
    return yt_video_filename


def clean_and_define_video_name(start_video_name: str) -> str:
    start_video_name = clean_string(start_video_name)
    start_video_name += ".mp4"
    now = datetime.now()
    date_time = now.strftime("%d,%m,%Y,,%H,%M,%S")
    start_video_name = f"date_time_{date_time}_title_{start_video_name}"
    return start_video_name


def full_pipe_line(data_dict: dict) -> None:
    yt_video_url, class_name = data_dict["url"], data_dict["face"]
    dataset_dir_path = "face_dataset"
    frames_base_dir = "frames"
    yt_video_save_dir = "."
    # download video
    yt_video_filename = download_yt_video(
        yt_video_url=yt_video_url,
        yt_video_save_dir=yt_video_save_dir,
    )
    # frames
    frame_dir = Path(frames_base_dir).joinpath(Path(yt_video_filename).stem)
    Path(frame_dir).mkdir(exist_ok=True, parents=True)
    frames_paths = video_to_frames(yt_video_filename, str(frame_dir))
    # rm video
    try:
        os.remove(yt_video_filename)
        Path(yt_video_filename).unlink()
    except FileNotFoundError:
        pass
    # faces
    face_dir = Path(dataset_dir_path).joinpath(class_name)
    face_dir.mkdir(exist_ok=True, parents=True)
    faces_paths = find_faces(frame_dir, face_dir)
    # rm frames
    shutil.rmtree(frame_dir)


if __name__ == "__main__":
    data_dicts = read_dict_from_csv("data.csv")
    for data_dict in data_dicts:
        full_pipe_line(data_dict)
