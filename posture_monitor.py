"""Seated posture monitor.

Reads a video source, tracks the upper body with MediaPipe Pose, classifies
posture over time and writes a JSON summary of the session.
"""
import argparse
import sys
import time

import cv2

EXIT_OK = 0
EXIT_SOURCE_ERROR = 3  # camera missing or file unreadable


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


def main():
    args = parse_args()
    cap = open_source(args.source)
    if cap is None:
        print(f"error: cannot open source {args.source!r}", file=sys.stderr)
        return EXIT_SOURCE_ERROR

    is_file = not args.source.isdigit()
    started = time.monotonic()
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_time(cap, is_file, started) > args.duration:
            break
        frames += 1
    cap.release()

    print(f"read {frames} frames")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
