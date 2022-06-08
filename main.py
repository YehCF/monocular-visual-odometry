import os
import errno
import argparse
import shutil
import yaml
import cv2

import numpy as np

from mvo import FlowBasedMonocularVisualOdometry

from helpers.video_utils import video_to_frames
from helpers.video_utils import load_frames


def main(file: str):
    """The main method to run for monocular visual odometry project

    Parameters
    ----------
    file : str
        the yaml configuration file for the target video & monocular visual odometry
    """

    # init & load configurations
    cfg = None

    with open(file, 'r') as f:

        cfg = yaml.load(f, yaml.Loader)

    # process video
    video_path = cfg['video']['video_path']

    if not os.path.isfile(video_path):

        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), video_path)

    # export video to frames
    video_to_frames(video_path)

    # get parent directory & load exported frames
    parent_dir = os.path.dirname(video_path)

    frames, fps = load_frames(parent_dir)

    # instantiate mvo
    mvo = FlowBasedMonocularVisualOdometry(**cfg['mvo'])

    # run visual odometry
    mvo.estimate(frames)

    # overlay / render with ... parameters
    mvo.overlay(frames)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run monocular visual odometry and overlay the target video with estimated trajectory')

    parser.add_argument(
        '-c', '--config', help='the directory of the configuration yaml file', required=True)

    args = parser.parse_args()

    # run the pipeline
    main(args.config)
