import cv2
import numpy as np

from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from tqdm.notebook import tqdm


class FlowBasedMonocularVisualOdometry:

    def __init__(self,
                 frame_height: int,
                 frame_width: int,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 max_disp_to_track: int = 5,
                 min_num_kps_to_track: int = 3000,
                 min_angle: int = 1,
                 max_angle: int = 120,
                 min_triangulated_points: int = 20,
                 max_frame_interval: int = 5,
                 track_window_size: tuple = (12, 12)):
        """The optical-flow-based monocular visual odometry method

        Parameters
        ----------
        frame_height : int
            the height of the given frames
        frame_width : int
            the width of the given frames
        fx : float
            the focal length along x axis
        fy : float
            the focal length along y axis
        cx : float
            the projection center x
        cy : float
            the projection center y
        max_disp_to_track : int, optional
            max displacement to track for the optical flow method, by default 5
        track_window_size : tuple, optional
            the patch size to track for the optical flow method, by default (12, 12)
        min_num_kps_to_track : int, optional
            min number of keypoints to track between the frames, by default 3000
        max_frame_interval : int, optional
            max interval for a reference / last frame of the current frame, by default 5
        min_angle : int, optional
            min angle for triangulated 3D points, by default 1
        max_angle : int, optional
            max angle for triangulated 3D points, by default 120
        min_triangulated_points : int, optional
            min number of triangulated points, by default 20

        """

        # build up camera intrinsics (K)
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

        self.focal = (fx + fy) / 2

        # (cx, cy)
        self.cxy = (cx, cy)

        # image (frame) height & width
        self.height = frame_height
        self.width = frame_width

        # init feature detector & params
        self.detector = cv2.FastFeatureDetector_create(
            threshold=10, nonmaxSuppression=True)

        # init feature tracker & params
        self.min_num_kps_to_track = min_num_kps_to_track

        self.track_window_size = track_window_size

        self.max_disp_to_track = max_disp_to_track

        self.lk_params = dict(winSize=self.track_window_size,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))

        self.max_frame_interval = max_frame_interval

        # the criteria for triangulability
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_triangulated_points = min_triangulated_points

    def detect_kps(self, frame: np.ndarray, cell_size: int = 128, max_per_cell: int = 150):
        """Detect keypoints in a given frame in an uniformly distributed manner

        Parameters
        ----------
        frame : np.ndarray
            the image in RGB, shape (height, width, 3).
        cell_size : int, optional
            the size of each cell, by default 128
        max_per_cell : int, optional
            max number of keypoints to detect of each cell, by default 150

        Returns
        -------
        np.ndarray
            the keypoint coordinates (x, y), shape (N, 2), N: the total number of keypoints
        """

        ht, wd = frame.shape[:2]

        kps = []

        # get number of cells along x & y axis
        xs, ys = wd // cell_size, ht // cell_size

        for ix in range(xs):

            for iy in range(ys):

                # top & bottom coordinate of a cell
                start_y, end_y = iy * cell_size, (iy + 1) * cell_size

                if iy == ys - 1:

                    end_y = ht - 1

                # left & right coordinate of a cell
                start_x, end_x = ix * cell_size, (ix + 1) * cell_size

                if ix == xs - 1:

                    end_x = wd - 1

                # extract the patch
                patch = frame[start_y:end_y, start_x:end_x]

                # detect & sort the keypoints by their importances
                patch_kps = sorted(self.detector.detect(
                    patch), key=lambda x: x.response, reverse=True)[:max_per_cell]

                # get the keypoint in (x, y) form
                for kp in patch_kps:

                    kps.append(
                        np.array([kp.pt[0] + start_x, kp.pt[1] + start_y]))

        return np.array(kps)

    def track_kps(self, last_frame, last_kps, curr_frame):
        """Track the keypoints using the Lucas-Kanade method

        Parameters
        ----------
        last_frame : [np.ndarray]
        last_kps : [np.ndarray], shape (N, 2)
        curr_frame : [np.ndarray]

        Returns
        -------
        curr_kps : [np.ndarray], shape (N, 2)
        """

        valid_idx = ~np.isnan(last_kps[:, 0])

        valid_curr_kps, st, err = cv2.calcOpticalFlowPyrLK(last_frame,
                                                           curr_frame,
                                                           last_kps[valid_idx].astype(
                                                               np.float32),
                                                           None,
                                                           **self.lk_params)

        # verification - 1 : error should be within the range
        st[(err > self.max_disp_to_track).squeeze()] = 0

        # verification - 2 : coordinate boundary [0, self.height - 1], [0, self.width - 1]
        st[(valid_curr_kps[:, 0] <= 0) | (valid_curr_kps[:, 1] <= 0)] = 0
        st[(valid_curr_kps[:, 0] >= self.width - 1) |
            (valid_curr_kps[:, 1] >= self.height - 1)] = 0

        # create the mask
        # => value 1: the ones should be masked out (large error or beyond the boundary)
        mask = (st == 0).squeeze()

        valid_curr_kps[mask] = np.nan

        curr_kps = np.ones_like(last_kps) * np.nan

        curr_kps[valid_idx] = valid_curr_kps

        return curr_kps

    def estimate_pose(self, last_kps, curr_kps):
        """Estimate the camera pose with epipolar geometry

        Parameters
        ----------
        last_kps : [np.ndarray], shape (N, 2)
        curr_kps : [np.ndarray], shape (N, 2)

        Returns
        -------
        R : [np.ndarray], shape (3, 3)
        T : [np.ndarray], shape (3, 1)
        all_inliers : [np.ndarray], shape (N,)
        """

        valid_idx = ~np.isnan(curr_kps[:, 0])

        # note:
        # mask : 0 & 255
        E, mask = cv2.findEssentialMat(last_kps[valid_idx],
                                       curr_kps[valid_idx],
                                       pp=self.cxy,
                                       focal=self.focal,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=0.1)

        # get the camera pose (of current frame)
        # inliers: 0 & 1
        _, R, T, inliers = cv2.recoverPose(E,
                                           last_kps[valid_idx],
                                           curr_kps[valid_idx],
                                           focal=self.focal,
                                           pp=self.cxy,
                                           mask=(mask * 255).astype(np.uint8))

        inliers = inliers.squeeze().astype(np.bool_)

        # back to original size
        all_inliers = (np.zeros(curr_kps.shape[0],)).astype(np.bool_)
        all_inliers[valid_idx] = inliers

        return R, T, all_inliers

    def triangulate_3d_landmarks(self,
                                 last_kps,
                                 last_cam_pose,
                                 curr_kps,
                                 curr_cam_pose,
                                 inliers=None):
        """Triangulate 3d points with the two views

        Parameters
        ----------
        last_kps : [np.ndarray], shape (N, 2)
        last_cam_pose : [np.ndarray], shape (3, 4)
            last_cam_poses[:3, :3] - the rotation matrix 
            last_cam_poses[:3, [3]] - the translation
        curr_kps : [np.ndarray], shape (N, 2)
        curr_cam_pose : [np.ndarray], shape (3, 4)
            curr_cam_poses[:3, :3] - the rotation matrix
            curr_cam_poses[:3, [3]] - the translation
        inliers : [np.ndarray], shape (N,)
            the inliers during the estimation of essential matrix

        Returns
        -------
        kps_3d : [np.ndarray], shape (N, 3)
            the triangulated 3d points
        """

        if inliers is None:

            inliers = np.ones((curr_kps.shape[0])).astype(np.bool_)

        # valid points able to be triangulated
        mask = inliers & (~np.isnan(curr_kps[:, 0]))

        if mask.sum() == 0:

            return None

        valid_last_kps = last_kps[mask]
        valid_curr_kps = curr_kps[mask]

        n_valids = valid_last_kps.shape[0]

        inv_K = np.linalg.inv(self.K)

        # normalized / calibrated points: (3, n_valids)
        norm_last_kps = inv_K @ (np.concatenate([valid_last_kps, np.ones((n_valids, 1))],
                                                axis=1).T)
        norm_curr_kps = inv_K @ (np.concatenate([valid_curr_kps, np.ones((n_valids, 1))],
                                                axis=1).T)

        valid_kps_homo = cv2.triangulatePoints(last_cam_pose,
                                               curr_cam_pose,
                                               norm_last_kps[:2],
                                               norm_curr_kps[:2])

        # homogeneous points to 3d points
        valid_kps_3d = cv2.convertPointsFromHomogeneous(valid_kps_homo.T)

        # make sure triangulation is good

        kps_3d = np.ones((curr_kps.shape[0], 3)) * np.nan
        kps_3d[mask] = valid_kps_3d.squeeze()

        kps_3d[kps_3d[:, -1] < 0] = np.nan

        return kps_3d

    def get_triangulation_angles(self, kps_3d, R, T):
        """Calculate the angles of triangulated points between the two bearing vectors.
        One from the point to the world center; the other from the point to the current camera.

        Parameters
        ----------
        kps_3d : [np.ndarray], shape (N, 3)
        R : [np.ndarray], shape (3, 3)
            the rotation matrix of the current frame (camera)
        T : [np.ndarray], shape (3, 1)
            the translation vector of the current frame (camera)

        Returns
        -------
        all_angles : [np.ndarray]
            the bearing angles of all 3d points
        """

        mask = ~np.isnan(kps_3d[:, 0])

        valid_kps_3d = kps_3d[mask]

        vec_in_ref = valid_kps_3d / \
            np.linalg.norm(valid_kps_3d, axis=1)[:, np.newaxis]

        vec_in_cam = (valid_kps_3d.T + R.T @ T).T
        vec_in_cam = vec_in_cam / \
            np.linalg.norm(vec_in_cam, axis=1)[:, np.newaxis]

        angles = np.arccos((vec_in_ref * vec_in_cam).sum(axis=1)) * 180

        all_angles = np.ones((kps_3d.shape[0],)) * np.nan
        all_angles[mask] = angles

        return all_angles

    def optimize(self):

        pass