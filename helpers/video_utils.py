import os
import cv2

from tqdm import tqdm


def video_to_frames(video_path: str, export_directory: str = None):
    """ Export video to frames, which creates a folder ('frames') in the same folder containing this video.
    Moreover, a txt file containing the frame rate is generated in the same folder.

    Parameters
    ----------
    video_path : str
        the path of the video

    """

    if export_directory is None:

        # use parent directory as export_directory
        export_directory = os.path.dirname(video_path)

    # open the video using opencv
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # write frame rate under the same parent directory
    with open(os.path.join(export_directory, "fps.txt"), "w") as f:

        f.write(str(fps))

    # create folder for frames if not exist
    # the folder `frame` is also under the same parent directory
    frame_folder = os.path.join(export_directory, "frames")

    if not os.path.isdir(frame_folder):

        os.makedirs(frame_folder)

    # export frames
    for i_frame in tqdm(range(nframes)):

        ret, frame = cap.read()

        # handle exeception (early None encountered)
        if not ret:
            break

        # write each frame into the folder
        cv2.imwrite(os.path.join(
            frame_folder, 'frame_%06d.jpg' % i_frame), frame)


def load_frames(video_folder: str, start_time: int, end_time: int):
    """ Load frames from the folder with specified start_time & end_time in seconds

    Parameters
    ----------
    video_folder : str
        the folder where the video's 'frames' subfolder is
    start_time : int
        the start time to load
    end_time : int
        the end time to load

    Returns
    -------
    frames : [str], shape (N,)
        a list of (sorted) filenames
    fps : float
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

    # check the start_time & end_time
    if start_time is None:

        start_time = 0

    if end_time is not None:

        start_fr = int(start_time * fps)
        end_fr = int(end_time * fps)

        frames = []

        print(f'loading video frames from {start_fr} to {end_fr} ... ')

        for i_frame in tqdm(range(start_fr, end_fr)):

            frame_path = os.path.join(
                video_folder, 'frames', 'frame_%06d.jpg' % i_frame)

            frames.append(frame_path)

    else:

        print(f'loading all the video frames ... ')
        # no end_time specified
        # use all the frames
        frames = sorted([os.path.join(video_folder, 'frames', fn) for fn in tqdm(os.listdir(
            os.path.join(video_folder, 'frames'))) if ".jpg" in fn])

    return frames, fps


def get_image(filename):
    """Read the image data from the filename using opencv

    Parameters
    ----------
    filename : str
        filename of the frame

    Returns
    -------
    np.ndarray
        shape (height, width, 3)
    """

    img = cv2.imread(filename)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
