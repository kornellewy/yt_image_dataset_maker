import os
import shutil
from pathlib import Path
from pprint import pprint

from pytube import YouTube

# read file with yt urls

# itertate over it

# cut is per frame

# try foir blurnes

# find all faces in images


def download_yt_video(yt_video_url, yt_video_save_dir, yt_video_filename):
    yt_video = YouTube(yt_video_url)
    max_resolution_tag = 0
    max_resolution = 0
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




if __name__ == "__main__":
    yt_video_url = "https://youtu.be/c1ZLS-dnw94"
    yt_video_save_dir = "."
    yt_video_filename = "1.mp4"
    download_yt_video(
        yt_video_url=yt_video_url,
        yt_video_save_dir=yt_video_save_dir,
        yt_video_filename=yt_video_filename,
    )
