"""Seated posture monitor.

Reads a video source, tracks the upper body with MediaPipe Pose, classifies
posture over time and writes a JSON summary of the session.
"""
import argparse
import json
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

EXIT_OK = 0
EXIT_SOURCE_ERROR = 3  # camera missing or file unreadable
EXIT_NO_PERSON = 4     # source worked but nobody was found in frame

# PROVISIONAL: a guess, to be replaced by a value taken from recorded data.
NECK_RATIO_BAD = 0.30  # below this the head is sunk towards the shoulders


def parse_args():
    parser = argparse.ArgumentParser(description="Seated posture monitor")
    parser.add_argument("--source", default="0",
                        help="webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--duration", type=float, default=60,
                        help="seconds of video to analyse")
    parser.add_argument("--out", default="posture.json",
                        help="where to write the JSON summary")
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

    def point(index):
        return np.array([lm[index].x * width, lm[index].y * height])

    left_shoulder, right_shoulder = point(Pose.LEFT_SHOULDER), point(Pose.RIGHT_SHOULDER)
    ear_mid = (point(Pose.LEFT_EAR) + point(Pose.RIGHT_EAR)) / 2
    shoulder_mid = (left_shoulder + right_shoulder) / 2

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    ears_above_shoulders = shoulder_mid[1] - ear_mid[1]  # image y grows downwards
    return float(ears_above_shoulders / shoulder_width)


def main():
    args = parse_args()
    cap = open_source(args.source)
    if cap is None:
        print(f"error: cannot open source {args.source!r}", file=sys.stderr)
        return EXIT_SOURCE_ERROR

    is_file = not args.source.isdigit()
    started = time.monotonic()
    frames = 0
    ratios = []
    labels = []  # True = good posture, one entry per frame with a person
    with mp.solutions.pose.Pose(model_complexity=1) as pose:  # 1 = full model, bundled with the package
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_time(cap, is_file, started) > args.duration:
                break
            frames += 1

            height, width = frame.shape[:2]
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                ratio = neck_ratio(result.pose_landmarks, width, height)
                ratios.append(ratio)
                labels.append(ratio >= NECK_RATIO_BAD)
    cap.release()

    if not ratios:
        print("error: nobody found in frame", file=sys.stderr)
        return EXIT_NO_PERSON

    # Naive version: every flip of the per-frame label counts as a break.
    breaks = sum(1 for prev, cur in zip(labels, labels[1:]) if prev and not cur)
    summary = {
        "frames_read": frames,
        "frames_with_person": len(ratios),
        "good_fraction": round(sum(labels) / len(labels), 3),
        "breaks": breaks,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
