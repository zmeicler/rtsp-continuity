# Check Continuity of RTSP Stream by Comparing Sampled Frames

An OpenCV-based pipeline that verifies RTSP stream continuity by checking for three conditions:

- **Image Corruption**: Determines if the second sampled frame is significantly blurrier than the first by comparing Laplacian variance. Also, checks for significant color loss by comparing the normalized histograms of color distribution.
- **Camera Movement**: Detects whether the camera has moved using ORB feature matching and homography on the sampled frames.
- **Obstructions**: Identifies any new obstructions appearing in specified regions of interest (ROIs) using contour analysis.

## Installation

Set up the Python environment using Conda:

```sh
conda create -n framechecker python=3.10
conda activate framechecker
pip install opencv-python numpy
```

## Running the Continuity Checker

Execute the script with the following command:

```sh
python continuity_checker.py /PATH/TO/BEFORE-IMAGE.png /PATH/TO/AFTER-IMAGE.png
```

Replace `/PATH/TO/BEFORE-IMAGE.png` and `/PATH/TO/AFTER-IMAGE.png` with the actual paths to your sampled frames.

