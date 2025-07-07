import os
import json
import time
import numpy as np
import cv2
from tqdm import tqdm

from utils import VideoUtils

# Configuration parameters
FEATURES_COUNT = 1000
FEATURES_QUALITY = 0.01
FEATURES_MIN_DISTANCE = 20
HOMOGRAPHY_RANSAC_THRESHOLD = 5.0
HOMOGRAPHY_MIN_FEATURES = 10
SMOOTHING_RADIUS = 50
VIDEO_INTERPOLATION_METHOD = cv2.INTER_LINEAR
VIDEO_BORDER_MODE = cv2.BORDER_REPLICATE

class VideoStabilizer(VideoUtils):
    def __init__(self):
        super().__init__()
    
    def apply_stabilization(self, inp, out, utils, smoothing_radius=SMOOTHING_RADIUS):
        """Apply video stabilization"""
        cap, n, w, h, fps = self.open_video_capture(inp)

        _, prev = cap.read()
        if prev is None:
            raise ValueError("Could not read first frame from video")
        
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        transforms = []

        # Motion estimation
        for i in tqdm(range(n - 1), desc="Motion estimation", leave=False, ncols=80):
            ok, curr = cap.read()
            if not ok:
                break
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            p0 = cv2.goodFeaturesToTrack(prev_gray, FEATURES_COUNT, FEATURES_QUALITY, FEATURES_MIN_DISTANCE)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

            if p0 is None or p1 is None or st is None:
                H = np.eye(3)
            else:
                st = st.flatten()
                p0, p1 = p0[st == 1], p1[st == 1]
                if len(p0) < HOMOGRAPHY_MIN_FEATURES:
                    H = np.eye(3)
                else:
                    H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, HOMOGRAPHY_RANSAC_THRESHOLD)
                    H = np.eye(3) if H is None else H

            transforms.append(H)
            prev_gray = curr_gray

        # Smooth trajectory
        cumulative = [np.eye(3)]
        for H in transforms:
            cumulative.append(H @ cumulative[-1])
        smooth = self.smooth_homographies(cumulative, smoothing_radius)

        # Write stabilized video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame0 = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(out, fourcc, fps, (w, h))
        vw.write(frame0)

        for i in tqdm(range(1, n), desc="Writing stabilized video", leave=False, ncols=80):
            ok, f = cap.read()
            if not ok:
                break
            Hcorr = smooth[i] @ np.linalg.inv(cumulative[i])
            stab = cv2.warpPerspective(
                f, Hcorr, (w, h), flags=VIDEO_INTERPOLATION_METHOD, borderMode=VIDEO_BORDER_MODE
            )
            vw.write(stab)

        cap.release()
        vw.release()
        utils.record_timing('time_to_stabilize')
    
    def moving_average_1d(self, x, radius):
        """1D moving average"""
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        return np.convolve(x, kernel, mode="same")

    def smooth_homographies(self, homographies, radius=50):
        """Smooth homography matrices using moving average"""
        smoothed = []
        for i in range(9):
            series = np.array([H.flatten()[i] for H in homographies])
            smoothed.append(self.moving_average_1d(series, radius))
        smoothed = np.stack(smoothed, axis=1).reshape((-1, 3, 3))
        return smoothed