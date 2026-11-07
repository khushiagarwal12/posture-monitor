# posture-monitor
# Seated Posture Monitor

A lightweight, CPU-based Python application that monitors seated posture using a webcam or video file. It uses MediaPipe Pose landmarks to estimate posture, applies temporal stabilisation to reduce false detections, and generates a JSON summary of the session.

## Features

- **Pose detection:** Uses MediaPipe Pose to track ear and shoulder landmarks.
- **Scale-invariant metric:** Calculates a neck ratio using the vertical distance between the ear midpoint and shoulder midpoint, normalised by shoulder width.
- **Posture classification:** Classifies posture as good or bad using configurable thresholds.
- **Temporal stabilisation:** Uses a rolling median, hysteresis and minimum dwell time to avoid classifying brief movements or noisy frames as posture breaks.
- **Session summary:** Generates a JSON report containing good-posture fraction, confirmed breaks and an overall score.
- **Headless execution:** Runs without a graphical interface; live preview is optional.

## Setup

Requires Python 3.10 or later.

```bash
git clone https://github.com/khushiagarwal12/posture-monitor.git
cd posture-monitor

python -m venv .venv
```

Activate the environment:

**Windows**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**Analyse a webcam session:**

```bash
python posture_monitor.py --source 0 --duration 45 --out posture.json
```

**Enable live preview and save frame-level measurements:**

```bash
python posture_monitor.py --source 0 --duration 45 --out posture.json --metrics-csv metrics.csv --preview
```

**Analyse a video file:**

```bash
python posture_monitor.py --source input.mp4 --out posture.json
```

The preview is optional. Press `q` to stop preview mode.

## Design Decisions

**Posture metric and thresholds**

The neck ratio is calculated as:

`(shoulder midpoint y - ear midpoint y) / shoulder width`

Pixel coordinates are used to account for image aspect ratio. Normalising by shoulder width reduces sensitivity to changes in camera distance.

The initial thresholds were calibrated using observed upright and slouched readings:

- Enter bad posture: `0.50`
- Return to good posture: `0.54`

These are empirical starting values for the tested camera setup, not universal ergonomic thresholds.

**Stabilisation**

Three mechanisms are used:

1. A 15-frame rolling median reduces the effect of isolated landmark fluctuations.
2. Hysteresis uses separate thresholds for entering and exiting bad posture.
3. A 2-second minimum dwell time ensures a posture change persists before it is confirmed.

This prevents brief movements and individual noisy frames from generating unnecessary posture breaks.

**Scoring**

The score is calculated as:

`score = 100 × good_fraction - 3 × confirmed_breaks`

The result is clamped between 0 and 100. The break penalty is a simple heuristic intended to account for the frequency of posture interruptions.

## Output

The JSON summary includes:

- `score`: Overall score between 0 and 100.
- `good_fraction`: Fraction of usable frames classified as good.
- `breaks`: Number of confirmed good-to-bad transitions.
- `breaks_per_minute`: Confirmed breaks normalised by session duration.
- `analysed_seconds`: Analysed session duration.
- `frames_read`: Total frames successfully read.
- `frames_with_person`: Frames with usable pose measurements.
- `raw_breaks_without_stabilising`: Transitions that would have occurred without stabilisation.

The optional CSV contains timestamps, raw and smoothed neck ratios, and classified posture states for calibration and inspection.

## Limitations

- Designed for a single seated person positioned in front of the camera.
- Relies on visible and sufficiently confident ear and shoulder landmarks.
- The geometric metric is a simplified posture proxy; it does not comprehensively assess spinal alignment or all forms of slouching.
- Thresholds may require recalibration for different camera positions, individuals and seating setups.
- The score is a heuristic indicator, not a medical or ergonomic assessment.
