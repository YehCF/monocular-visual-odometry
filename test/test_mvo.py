import os
import yaml
import numpy as np

from mvo import FlowBasedMonocularVisualOdometry
from helpers.video_utils import load_frames
from helpers.video_utils import get_image


def test_load_frames():

    frames, fps = load_frames("images", None, None)

    assert len(frames) == 3, 'There should be 3 frames !'
    assert abs(fps - 20) <= 1e-1, 'fps should be 20.000982999884354 !'


def test_flow_based_mvo():
    """Test the basic functionality of flow-based monocular odometry method with the test images.
    The translation should be close to 0 in this case.
    """

    cfg = None

    with open('config/default.yaml', 'r') as f:

        cfg = yaml.load(f, yaml.Loader)

    frames, _ = load_frames("images", None, None)

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

    est_R = np.eye(3)
    est_T = np.zeros((3, 1))

    for R, T in mvo.poses:

        est_R = est_R @ R.T
        est_T += R.T @ (-T)

    assert abs(1 - np.linalg.det(est_R)
               ) <= 1e-3, 'The determinant of the R must be close to 1 !'
    assert abs(1 - np.linalg.norm(est_T)
               ) <= 1 + 1e-5, 'The camera translation should be between 0 and 1.'
