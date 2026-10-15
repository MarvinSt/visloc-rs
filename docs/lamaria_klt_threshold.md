# LaMAria: KLT forward-backward threshold

Basalt's optical flow tracks each patch from the previous frame to the new
one, tracks it back, and drops the track when the point it returns to lies
more than √`config.optical_flow_max_recovered_dist2` pixels from where it
started. The value in every LaMAria config is Basalt's EuRoC default, 0.04
(0.2 px).

## Why it matters in low light

sequence_4_10 (Low light) is recorded at the 14 ms exposure ceiling and
gain 15. Its images have a median intensity of 8 out of 255, with visible
sensor noise and motion blur. A proxy of the frontend on 40 frames of
sequence_4_10 (FAST in 50 px cells with Basalt's threshold schedule, OpenCV
pyramidal LK, round trip as above) keeps about 32 of 104 detections per frame
at 0.2 px and 43 at 0.5 px. A light Gaussian blur and contrast-limited
histogram equalization (CLAHE) before tracking raise the detections to 165
per frame, but the tracked count only to 51 at 0.5 px.

## Measured

Setup: the submission defaults (rectified IMU, per-device refinement,
multicam big window, `gyro_bias_std` 1e-7, online mapper with the 5° loop
gate), only `config.optical_flow_max_recovered_dist2` changed from 0.04 to
0.25 (0.5 px).

| Sequence | Track | VIO: 0.2 px → 0.5 px | Mapper: 0.2 px → 0.5 px | VIO Pose R@1m / R@5m |
|---|---|---|---|---|
| sequence_4_10 | Low light | 34.49 → **39.00** | 31.62 → 36.65 | 5.5 / 86.5 % → **14.1 / 98.6 %** |
| sequence_1_19 | Short | 71.96 → **73.79** | 68.54 → 70.08 | 76.3 / 100 % → 74.5 / 100 % |

On sequence_4_10 the RMSE over separately aligned 60 s segments falls from
0.442 m to 0.381 m. The VIO takes about 11 % longer, since more tracks
survive.

Status: better on both tracks measured so far; Medium, Long and Moving
platform are still to be measured before it becomes the default.
