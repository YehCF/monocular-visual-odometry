# monocular-visual-odometry
This repository contains the monocular visual odometry algorithms for DTA. The main task of this repo is to estimate the ego trajectory with respective to the first frame of the video and then render the future trajectory onto the reference frames, which helps annotators in DTA tasks.

### Install
---
* `$git@github.com:PerceptiveAutomata/monocular-visual-odometry.git`
* `$conda activate perceptive3`
  *  no additional packages needed beyond perceptive3

### Video Preparation
---
* Select a video (.mp4) to process

### Steps
---
This pipeline takes in a yaml configuration file. There is an example in `config`, called `config/default.yaml`
- copy the `config/default.yaml` and rename it [Optional]
- change the video path in the yaml file
- change the camera intrinsics if needed
- run the following command:
  - `$python main.py -c [your yaml file]`
  - for example `$python main.py -c config/default.yaml`

### Results
---
- A new folder called `odometry_[video name]` is created in the same folder containing the video. There are 5 things in the generated folder:
  1. `frames` : the folder, containing all the original frames from the video
  2. `fps.txt` : the fps of the video
  3. `mvo.hdf5` : the estimated camera poses along with camera intrinsics
  4. `overlaid-[video name].mp4` : the video (specified between the start_time and the end_time) with estimated trajectory
  5. `[video name]-overlaid-frames` : all the frames with estimated trajectory


### Demo
---
- The examples can be found [here](https://drive.google.com/drive/u/3/folders/1jgyRffiWhCtqDzBIIqBxWkthTDX22r0A)


### Test Cases
---
- $pytest
