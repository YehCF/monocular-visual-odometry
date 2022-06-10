import os
import errno
import argparse
import shutil
import yaml
import subprocess
import numpy as np

from tqdm import tqdm

from mvo import FlowBasedMonocularVisualOdometry

from helpers.video_utils import video_to_frames
from helpers.video_utils import load_frames
from helpers.video_utils import get_image
from helpers.reproject_utils import get_projected_trajectory_from_poses
from helpers.reproject_utils import export_projected_frame
from helpers.reproject_utils import check_trajectory


def main(config: str):
    """The main method to run for monocular visual odometry project.
    A new folder named after the video name is created to store all the frames & generated videos.

    Parameters
    ----------
    config : str
        the yaml configuration file for the target video & monocular visual odometry
    """

    # init & load configurations
    cfg = None

    with open(config, 'r') as f:

        cfg = yaml.load(f, yaml.Loader)

    # process video
    video_path = cfg['video']['video_path']

    # get parent directory & load exported frames
    parent_dir = os.path.dirname(video_path)

    video_fn = video_path.split("/")[-1].split(".")[0]

    if not os.path.isfile(video_path):

        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), video_path)

    # create target folder
    target_folder = os.path.join(parent_dir, f'odometry_{video_fn}')

    if not os.path.isdir(target_folder):

        os.makedirs(target_folder)

    if not os.path.isdir(os.path.join(target_folder, "frames")):

        # export video to frames
        video_to_frames(video_path, target_folder)

    frames, fps = load_frames(target_folder, start_time=cfg['video'].get(
        'start_time', None), end_time=cfg['video'].get('end_time', None))

    # get image height & width
    frame_height, frame_width = get_image(frames[0]).shape[:2]

    # add height & width info into cfg['mvo']
    cfg['mvo']['frame_height'] = frame_height
    cfg['mvo']['frame_width'] = frame_width

    # instantiate mvo
    mvo = FlowBasedMonocularVisualOdometry(**cfg['mvo'])

    # run visual odometry
    print(f'Estimating the camera poses ... ')
    mvo.estimate(frames)

    # poses : [[R, T], ...]
    poses = mvo.get_smoothed_poses()

    # check poses
    check_trajectory(poses, fps)

    # render & export the frame
    export_directory = os.path.join(
        target_folder, f'{video_fn}-overlay-frames')

    if os.path.isdir(export_directory):

        shutil.rmtree(export_directory)

    os.makedirs(export_directory)

    print(f'Export overlaid frames ... ')

    for i_frame in tqdm(range(len(frames) - int(fps))):

        end_frame = i_frame + 10 * int(fps)

        projected_trajectory = get_projected_trajectory_from_poses(poses[i_frame:end_frame].copy(),
                                                                   mvo.K)

        export_projected_frame(i_frame,
                               get_image(frames[i_frame]),
                               projected_trajectory,
                               export_directory=export_directory)

    # run ffmpeg to make the video
    os.chdir(export_directory)
    subprocess.call(['ffmpeg', '-framerate',
                    f'{fps}', '-i', 'frame_%06d.jpg', f'../overlaid-{video_fn}.mp4'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run monocular visual odometry and overlay the target video with estimated trajectory')

    parser.add_argument(
        '-c', '--config', help='the directory of the configuration yaml file', required=True)

    args = parser.parse_args()

    # run the pipeline
    main(args.config)
