"""Seated posture monitor.

Reads a video source, tracks the upper body with MediaPipe Pose, classifies
posture over time and writes a JSON summary of the session.
"""
import argparse
import csv
import json
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

EXIT_OK = 0
EXIT_SOURCE_ERROR = 3  # camera missing or file unreadable
EXIT_NO_PERSON = 4     # source worked but too little usable data (nobody in frame)

MAX_CONSECUTIVE_BAD_READS = 30  # give up on the source after this many failed reads
MIN_LANDMARK_VISIBILITY = 0.5   # ignore a pose if ears/shoulders are not clearly seen
MIN_USABLE_FRAMES = 30          # fewer measured frames than this is not an observation
MIN_USABLE_FRACTION = 0.5       # ...nor is a person seen in under half of the frames

# Score = % of time in good posture, minus a flat penalty for each break.
# The penalty is a judgement call: it makes 5 minutes of 90% good with 1 break
# score higher than 90% good with 10 breaks.
POINTS_LOST_PER_BREAK = 3

# PROVISIONAL: guesses, to be replaced by values taken from recorded data.
# Hysteresis: it takes a lower value to ENTER "bad" than to get back to "good",
# so a reading hovering near one line cannot make the state flip back and forth.
NECK_RATIO_ENTER_BAD = 0.30  # smoothed ratio must fall below this to become bad
NECK_RATIO_EXIT_BAD = 0.33   # ...and rise above this to become good again
SMOOTHING_FRAMES = 15        # rolling-median window (~0.5 s at 30 fps)
MIN_DWELL_SECONDS = 2.0      # a change must hold this long before it counts


def parse_args():
    parser = argparse.ArgumentParser(description="Seated posture monitor")
    parser.add_argument("--source", default="0",
                        help="webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--duration", type=float, default=60,
                        help="seconds of video to analyse")
    parser.add_argument("--out", default="posture.json",
                        help="where to write the JSON summary")
    parser.add_argument("--preview", action="store_true",
                        help="show a live window (press q to stop); off by default")
    parser.add_argument("--metrics-csv",
                        help="also write every reading to this CSV, for choosing thresholds")
    return parser.parse_args()


def open_source(source):
    """Open a webcam (numeric source) or a video file. Returns None on failure."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        return None
    return cap


def frame_time(cap, is_file, started):
    """Seconds into the session.

    Files use the video's own clock (so a file is analysed faster than real
    time but timed correctly); webcams use wall-clock time.
    """
    if is_file:
        return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    return time.monotonic() - started


def neck_ratio(landmarks, width, height):
    """Scale-invariant posture metric for a camera facing the person.

    = (height of the ears above the shoulder line) / (shoulder width)

    Slouching drops the head towards the shoulders, so the ratio shrinks.
    Both distances scale equally with camera distance, so the ratio does not
    change if the person sits closer. Landmarks are converted to pixels first
    so the image aspect ratio does not distort the vertical/horizontal mix.
    """
    lm = landmarks.landmark
    Pose = mp.solutions.pose.PoseLandmark

    needed = [Pose.LEFT_EAR, Pose.RIGHT_EAR, Pose.LEFT_SHOULDER, Pose.RIGHT_SHOULDER]
    if min(lm[i].visibility for i in needed) < MIN_LANDMARK_VISIBILITY:
        return None  # MediaPipe guesses hidden joints; do not trust those guesses

    def point(index):
        return np.array([lm[index].x * width, lm[index].y * height])

    left_shoulder, right_shoulder = point(Pose.LEFT_SHOULDER), point(Pose.RIGHT_SHOULDER)
    ear_mid = (point(Pose.LEFT_EAR) + point(Pose.RIGHT_EAR)) / 2
    shoulder_mid = (left_shoulder + right_shoulder) / 2

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width < 1:
        return None  # degenerate pose, avoids dividing by ~0
    ears_above_shoulders = shoulder_mid[1] - ear_mid[1]  # image y grows downwards
    return float(ears_above_shoulders / shoulder_width)


class PostureStabiliser:
    """Turns noisy per-frame ratios into a steady good/bad state.

    Three layers, each fixing a different kind of noise:
      1. rolling median  - ignores single-frame landmark glitches (outliers)
      2. hysteresis      - two thresholds, so values near the line do not flicker
      3. minimum dwell   - the change must persist, so brief movements
                           (a cough, reaching for water) do not count
    """

    def __init__(self):
        self.recent = deque(maxlen=SMOOTHING_FRAMES)
        self.state = None           # "good" or "bad"; set from the first reading
        self.change_started = None  # when the ratio first crossed the line
        self.breaks = 0             # number of confirmed good -> bad changes

    def update(self, ratio, t):
        self.recent.append(ratio)
        smoothed = float(np.median(self.recent))

        if self.state is None:  # first reading: start in whatever state it is in
            self.state = "bad" if smoothed < NECK_RATIO_ENTER_BAD else "good"
            return smoothed, self.state

        if self.state == "good":
            crossed = smoothed < NECK_RATIO_ENTER_BAD
        else:
            crossed = smoothed > NECK_RATIO_EXIT_BAD

        if not crossed:
            self.change_started = None  # it came back, so forget the attempt
        elif self.change_started is None:
            self.change_started = t
        elif t - self.change_started >= MIN_DWELL_SECONDS:
            self.state = "bad" if self.state == "good" else "good"
            self.change_started = None
            if self.state == "bad":
                self.breaks += 1
        return smoothed, self.state


def measure_frame(pose, frame):
    """neck_ratio for one frame, or None if the frame is unusable."""
    if frame is None or frame.size == 0:
        return None
    try:
        height, width = frame.shape[:2]
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None
        return neck_ratio(result.pose_landmarks, width, height)
    except (cv2.error, ValueError):
        return None  # a corrupt frame is skipped, never fatal


def show_preview(frame, smoothed, state):
    """Draw the current reading on the frame. Returns False if q was pressed."""
    colour = (0, 200, 0) if state == "good" else (0, 0, 255)
    cv2.putText(frame, f"{state}  neck_ratio={smoothed:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.imshow("posture", frame)
    return cv2.waitKey(1) & 0xFF != ord("q")


def main():
    args = parse_args()
    cap = open_source(args.source)
    if cap is None:
        print(f"error: cannot open source {args.source!r}", file=sys.stderr)
        return EXIT_SOURCE_ERROR

    is_file = not args.source.isdigit()
    started = time.monotonic()
    frames = 0
    bad_reads = 0  # consecutive failed reads
    stabiliser = PostureStabiliser()
    good_frames = 0
    used_frames = 0
    raw_breaks = 0  # what a naive per-frame threshold would have counted
    previous_raw_good = None
    last_t = 0.0
    preview = args.preview
    csv_file = open(args.metrics_csv, "w", newline="") if args.metrics_csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(["t", "raw_ratio", "smoothed_ratio", "state"])
    with mp.solutions.pose.Pose(model_complexity=1) as pose:  # 1 = full model, bundled with the package
        while True:
            ok, frame = cap.read()
            if not ok:
                # A dropped frame: tolerate a few in a row. At the end of a file
                # the reads keep failing, so this also ends the loop.
                bad_reads += 1
                if bad_reads >= MAX_CONSECUTIVE_BAD_READS:
                    break
                time.sleep(0.02)
                continue
            bad_reads = 0

            t = frame_time(cap, is_file, started)
            if t > args.duration:
                break
            frames += 1

            ratio = measure_frame(pose, frame)
            if ratio is None:
                continue  # no person / low confidence: skip, keep the last state
            smoothed, state = stabiliser.update(ratio, t)
            last_t = t

            if csv_writer:
                csv_writer.writerow([round(t, 3), round(ratio, 4), round(smoothed, 4), state])
            if preview:
                try:
                    if not show_preview(frame, smoothed, state):
                        break
                except cv2.error:  # e.g. headless OpenCV build: carry on without it
                    print("warning: preview unavailable, continuing headless", file=sys.stderr)
                    preview = False

            used_frames += 1
            good_frames += state == "good"
            raw_good = ratio >= NECK_RATIO_ENTER_BAD
            raw_breaks += previous_raw_good is True and not raw_good
            previous_raw_good = raw_good
    cap.release()
    if csv_file:
        csv_file.close()
    if args.preview:
        cv2.destroyAllWindows()

    if frames == 0:
        print("error: source opened but no frame could be read", file=sys.stderr)
        return EXIT_SOURCE_ERROR
    if used_frames < MIN_USABLE_FRAMES or used_frames / frames < MIN_USABLE_FRACTION:
        print(f"error: person usable in only {used_frames} of {frames} frames", file=sys.stderr)
        return EXIT_NO_PERSON

    good_fraction = good_frames / used_frames
    minutes = max(last_t, 1) / 60
    score = 100 * good_fraction - POINTS_LOST_PER_BREAK * stabiliser.breaks
    summary = {
        "score": round(min(100, max(0, score))),
        "good_fraction": round(good_fraction, 3),
        "breaks": stabiliser.breaks,
        "breaks_per_minute": round(stabiliser.breaks / minutes, 2),
        "analysed_seconds": round(last_t, 1),
        "frames_read": frames,
        "frames_with_person": used_frames,
        "raw_breaks_without_stabilising": raw_breaks,  # shows what smoothing removed
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
