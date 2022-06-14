import os
import cv2
import numpy as np


def get_trajectory_from_poses(poses: list,
                              camera_offset=None):
    """Estimate the trajectory with respective to the given first camera pose

    Input:
        poses: [[R, T]], R => (3, 3), T => (3, 1)
        camera_offset: [np.ndarray], shape (3, 1)

    Parameters
    ----------
    poses : [[R, T]], R (3, 3), T (3, 1)
        a list of camera poses
    camera_offset : [float, float, float], optional
        the offsets along x, y, z axis for the camera center, by default None

    Returns
    -------
    np.ndarray
        the coordinates of the trajectory in (x, y, z) with respective to the reference frame (poses[0])
    """

    trajectory = []

    if camera_offset is None:

        # simulation for the trajectory on the road
        camera_offset = np.array([[0.5], [4.0], [16.0]])

    base_vector = poses[0][0].T @ (-poses[0][1])

    for R, T in poses[1:]:

        trajectory.append(
            (poses[0][0] @ (R.T @ (-T + camera_offset) - base_vector)).squeeze())

    return np.array(trajectory)


def get_projected_trajectory_from_poses(poses: list,
                                        K: np.ndarray,
                                        height: int = 1080,
                                        width: int = 1920,
                                        bounded: bool = True,
                                        camera_offset: np.ndarray = None):
    """Get the trajectory able to be back projected onto the image (reference frame, the first camera pose)

    Parameters
    ----------
    poses : list
        a list of camera poses [[R, T]]
    K : np.ndarray
        camera intrinsics
    height : int, optional
        the height of the frame, by default 1080
    width : int, optional
        the width of the frame, by default 1920
    bounded : bool, optional
        remove trajectory out of the image coordinate or not, by default True
    camera_offset : np.ndarray, optional
        the offsets along x, y, z axis for the camera center, by default None

    Returns
    -------
    np.ndarray
        the coordinates of the backprojected trajectory with respective to the reference frame (poses[0])
    float
        the smoothness (mean number of jerks) in this projected trajectory
    """

    trajectory = get_trajectory_from_poses(
        poses, camera_offset=camera_offset)

    projected_trajectory = (K @ trajectory.T).T

    projected_trajectory = projected_trajectory[:,
                                                :2] / (projected_trajectory[:, [2]] + 1e-10)

    # calculate the jerk of the whole projected trajectory
    mean_jerk = quantify_projected_jerk(projected_trajectory)

    if bounded:

        pt = []

        for i in range(projected_trajectory.shape[0]):

            if projected_trajectory[i][0] < 0 or projected_trajectory[i][0] >= width or projected_trajectory[i][1] < 0 or projected_trajectory[i][1] >= height:

                break

            pt.append(projected_trajectory[i])

        projected_trajectory = np.array(pt)

    return projected_trajectory, mean_jerk


def export_projected_frame(i_frame: int,
                           frame: np.ndarray,
                           projected_trajectory: np.ndarray,
                           n_jerk: int = -1,
                           export_directory: str = "temp",
                           from_color: tuple = (255, 51, 51),
                           to_color: tuple = (255, 204, 204),
                           from_thickness: int = 25,
                           to_thickness: int = 5,
                           jerk_threshold=10):
    """export the frame with backprojected trajectory on it

    Parameters
    ----------
    i_frame : int
        the index of the frame
    frame : np.ndarray
        the frame, shape (height, width, 3)
    projected_trajectory : np.ndarray
        the coordinates, shape (N, 2)
    n_jerk : int, optional
        the jerk of the trajectory with respect to this frame
    export_directory : str, optional
        the directory to export the overlaid frame, by default "temp"
    from_color : tuple, optional
        the color for the trajectory, by default (255, 51, 51)
    to_color : tuple, optional
        the color for the trajectory, by default (255, 204, 204)
    from_thickness : int, optional
        the thickness of the trajectory, by default 25
    to_thickness : int, optional
        the thickness of the trajectory, by default 5
     jerk_threshold : int, optional
        the threshold for jerk, by default 10.
        if jerk is greater than (& equal to) the threshold, the color would be turned into gray
    """

    # make the image to export
    export_img = frame.copy()

    pts = projected_trajectory.astype(np.int32)

    from_color = np.array(from_color)

    line_img = np.zeros_like(frame).astype(np.uint8)

    to_color = np.array(to_color)

    if n_jerk >= jerk_threshold:

        from_color = np.array([105, 105, 105]).astype(np.uint8)

        to_color = np.array([192, 192, 192]).astype(np.uint8)

    # get the color & thickness for each point
    for ipt in range(1, len(pts)):

        pt_color = (1 - (ipt / len(pts))) * from_color + \
            (ipt / len(pts)) * to_color

        pt_tk = int((1 - (ipt / len(pts))) * from_thickness +
                    (ipt / len(pts)) * to_thickness)

        cv2.polylines(line_img, [pts[ipt - 1: ipt + 1]],
                      False, pt_color, thickness=pt_tk)

    mask = (line_img > 0).astype(np.bool_)

    export_img[mask] = cv2.addWeighted(export_img, 0.4, line_img, 0.6, 0)[mask]

    export_img = cv2.cvtColor(export_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(export_directory, 'frame_%06d.jpg' %
                i_frame), export_img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def quantify_jerk(poses: np.ndarray, fps: float):
    """Estimate the jerk of the trajectory in a sliding window manner.
    This quantizes the number of jerks in each time window (10s).
    Parameters
    ----------
    poses : [[R, T]]
        the poses, each pose is the camera pose
    fps : float
        the fps of this video

    Returns
    -------
    list[int]
        the number of jerk in each window, shape (len(poses),)
    """

    trajectory = []

    for R, T in poses:

        trajectory.append(R.T @ (-T))

    # (N, 3, 1) to (N, 3)
    trajectory = np.array(trajectory)[..., 0]

    n_jerks = []

    window = 2

    duration = int(fps) * window

    for idx in range(max(1, len(trajectory) - duration)):

        tj = np.linalg.norm(trajectory[idx: idx + duration], axis=1)

        # jerk
        jerk = np.gradient(np.gradient(np.gradient(tj))) > 0

        n_jerk = jerk[1:] != jerk[:-1]

        n_jerks.append(n_jerk.sum())

    print(
        f'average number of jerks in 3D trajectory : {np.mean(n_jerks)} (n) per second & standard deviation : {np.std(n_jerks)} ! ')

    # padding the jerk
    n_pads = len(poses) - len(n_jerks)

    n_jerks += [n_jerks[-1]] * n_pads

    return n_jerks


def quantify_projected_jerk(projected_trajectory):
    """Estimate the jerk of the projected trajectory

    Parameters
    ----------
    projected_trajectory : np.ndarray, 
        the projected trajectory, shape (N, 2)

    Returns
    -------
    float
        mean number of jerk 
    """

    # root mean square
    traj = np.linalg.norm(projected_trajectory, axis=1)

    # calculate jerk (3rd derivative)
    jerk = np.gradient(np.gradient(np.gradient(traj)))

    # how many times it passes through the line (y=threshold) (pos to neg)
    threshold = 0.15

    jerk = (jerk >= threshold)

    mean_jerk = (jerk[1:] != jerk[:-1]).mean()

    return mean_jerk
