# PyTorch on Jetson

Real-time computer vision demos optimised for NVIDIA Jetson devices using [pixi](https://pixi.sh) for dependency management.

## Demos

| Script | Task | Output |
|---|---|---|
| `pose_cv2.py` | Human pose estimation | Local OpenCV window |
| `pose_ros2.py` | Human pose estimation | ROS 2 image topic → RViz |
| `segment_webcam.py` | Semantic segmentation | Local OpenCV window |

Pose estimation uses [YOLOv8n-pose](https://docs.ultralytics.com/tasks/pose/)..

## Requirements

- NVIDIA Jetson (tested on Jetson Nano / Orin)
- [pixi](https://pixi.sh) — handles all Python, CUDA, and ROS 2 dependencies

## Quick start

```bash
# Install pixi (once)
curl -fsSL https://pixi.sh/install.sh | sh

# Clone this repo and run demos
pixi run pose        # pose estimation → cv2 window
pixi run pose-ros2   # pose estimation → ROS 2 topic
pixi run segment     # segmentation → cv2 window
```

Model weights are downloaded automatically on first run.

## Options

### `pose_cv2.py` / `pose_ros2.py`

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index |
| `--width` | `640` | Capture width |
| `--height` | `480` | Capture height |
| `--score` | `0.5` | Detection confidence threshold |
| `--infer-width` | `320` | Inference resolution (smaller = faster, multiple of 32) |
| `--model` | `yolov26n-pose.pt` | YOLO model weights |

`pose_ros2.py` also accepts `--fps` (default `30`) to set the publish rate.

### ROS 2 visualisation

```bash
# In a separate terminal
pixi run rviz2
# Add → Image display → topic: /pose_estimation/image
```
