import os
import cv2

from tqdm import tqdm


def video_to_frames(video_path: str):
    """ Export video to frames, which creates a folder ('frames') in the same folder containing this video.
    Moreover, a txt file containing the frame rate is generated in the same folder.

    Parameters
    ----------
    video_path : [str]
        the path of the video

    """

    # get parent directory
    parent_path = os.path.dirname(video_path)

    # open the video using opencv
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # write frame rate under the same parent directory
    with open(os.path.join(parent_path, "fps.txt"), "w") as f:

        f.write(str(fps))

    # create folder for frames if not exist
    # the folder `frame` is also under the same parent directory
    frame_folder = os.path.join(parent_path, "frames")

    if not os.path.isdir(frame_folder):

        os.makedirs(frame_folder)

    # export frames
    for i_frame in tqdm(range(nframes)):

        ret, frame = cap.read()

        # handle exeception (early None encountered)
        if not ret:
            break

        # write each frame into the folder
        cv2.imwrite(os.path.join(frame_folder, f'frame_{i_frame}.jpg'), frame)


def load_frames(video_folder: str, start_time: int, end_time: int):
    """ Load frames from the folder with specified start_time & end_time in seconds

    Parameters
    ----------
    video_folder : [str]
        the folder where the video's 'frames' subfolder is
    start_time : [int]
        the start time to load
    end_time : [int]
        the end time to load

    Returns
    -------
    frames : [list[np.ndarray]]
        a list of loaded frames
    fps : [float]
        the original frames per second of the video
    """

    # get frame rate
    fr_file = os.path.join(video_folder, 'fps.txt')

    fps = None

    with open(fr_file, 'r') as f:

        fps = float(f.read())

        print(f'Video - {video_folder.split("/")[-1]} - fps: {fps}')

    # no fps.txt in the folder => probably haven't exported yet
    if not fps:
        raise ValueError('No fps is found!')

    start_fr = int(start_time * fps)
    end_fr = int(end_time * fps)

    frames = []

    print(f'loading video frames from {start_fr} to {end_fr}')

    for i_frame in tqdm(range(start_fr, end_fr)):

        frame_path = os.path.join(
            video_folder, 'frames', f'frame_{i_frame}.jpg')

        img = cv2.imread(frame_path)

        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return frames, fps
