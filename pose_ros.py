"""
Real-time Human Pose Estimation using YOLOv8-pose.
Publishes annotated image to a ROS 2 topic for visualization in RViz.

Usage:
    python pose_ros2.py [--camera 0] [--width 640] [--height 480]

RViz: Add an Image display subscribing to /pose_estimation/image
"""

import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO

# COCO keypoint skeleton connections
SKELETON = [
    (0, 1), (0, 2),           # nose -> eyes
    (1, 3), (2, 4),           # eyes -> ears
    (5, 6),                   # shoulders
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # shoulders -> hips
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

# BGR colors (cv2 uses BGR)
PART_COLORS = {
    "head":  ( 50, 200, 255),
    "arm":   (255, 200,  50),
    "torso": (120, 255,  80),
    "leg":   (150,  80, 255),
}

KEYPOINT_COLORS = [
    PART_COLORS["head"],    # nose
    PART_COLORS["head"],    # left_eye
    PART_COLORS["head"],    # right_eye
    PART_COLORS["head"],    # left_ear
    PART_COLORS["head"],    # right_ear
    PART_COLORS["torso"],   # left_shoulder
    PART_COLORS["torso"],   # right_shoulder
    PART_COLORS["arm"],     # left_elbow
    PART_COLORS["arm"],     # right_elbow
    PART_COLORS["arm"],     # left_wrist
    PART_COLORS["arm"],     # right_wrist
    PART_COLORS["torso"],   # left_hip
    PART_COLORS["torso"],   # right_hip
    PART_COLORS["leg"],     # left_knee
    PART_COLORS["leg"],     # right_knee
    PART_COLORS["leg"],     # left_ankle
    PART_COLORS["leg"],     # right_ankle
]

SKELETON_COLORS = [
    PART_COLORS["head"],    # nose-eye pairs
    PART_COLORS["head"],
    PART_COLORS["head"],
    PART_COLORS["head"],
    PART_COLORS["torso"],   # shoulders
    PART_COLORS["arm"],     # left arm
    PART_COLORS["arm"],
    PART_COLORS["arm"],     # right arm
    PART_COLORS["arm"],
    PART_COLORS["torso"],   # shoulders->hips
    PART_COLORS["torso"],
    PART_COLORS["torso"],   # hips
    PART_COLORS["leg"],     # left leg
    PART_COLORS["leg"],
    PART_COLORS["leg"],     # right leg
    PART_COLORS["leg"],
]


def run_inference(model, frame_bgr, score_thresh=0.5, infer_width=320):
    """Run YOLOv8-pose on a single BGR frame, return detections."""
    results = model(frame_bgr, imgsz=infer_width, conf=score_thresh, verbose=False)
    r = results[0]

    if r.keypoints is None or len(r.keypoints) == 0:
        return np.zeros((0, 4)), np.zeros((0, 17, 2)), np.zeros((0, 17)), np.zeros(0)

    boxes  = r.boxes.xyxy.cpu().numpy()       # [N, 4]
    scores = r.boxes.conf.cpu().numpy()       # [N]
    kp_xy  = r.keypoints.xy.cpu().numpy()    # [N, 17, 2]
    kp_conf = (
        r.keypoints.conf.cpu().numpy()        # [N, 17]
        if r.keypoints.conf is not None
        else np.ones((len(kp_xy), 17), dtype=np.float32)
    )

    return boxes, kp_xy, kp_conf, scores


def draw_poses(frame_bgr, boxes, kp_xy, kp_conf, scores):
    """Draw skeleton overlays onto frame in-place."""
    for person_idx, (xy, conf, score) in enumerate(zip(kp_xy, kp_conf, scores)):
        # Skeleton lines
        for conn_idx, (a, b) in enumerate(SKELETON):
            if conf[a] > 0.5 and conf[b] > 0.5:
                pt_a = (int(xy[a, 0]), int(xy[a, 1]))
                pt_b = (int(xy[b, 0]), int(xy[b, 1]))
                cv2.line(frame_bgr, pt_a, pt_b, SKELETON_COLORS[conn_idx], 2, cv2.LINE_AA)

        # Keypoint circles
        for i in range(17):
            if conf[i] > 0.5:
                cv2.circle(frame_bgr, (int(xy[i, 0]), int(xy[i, 1])), 5, KEYPOINT_COLORS[i], -1, cv2.LINE_AA)

        # Confidence label near bounding box
        if person_idx < len(boxes):
            x1, y1 = int(boxes[person_idx][0]), int(boxes[person_idx][1])
            cv2.putText(frame_bgr, f"person {person_idx}  {score:.2f}", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


class PosePublisher(Node):
    def __init__(self, args):
        super().__init__("pose_estimation")

        self.pub_image = self.create_publisher(Image, "/pose_estimation/image", 10)
        self.bridge = CvBridge()

        self.get_logger().info(f"Loading {args.model}...")
        self.model = YOLO(args.model)
        self.get_logger().info("Model ready.")

        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.camera}")

        self.score_thresh = args.score
        self.infer_width = args.infer_width

        # Timer drives the capture+inference+publish loop
        timer_period = 1.0 / args.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(
            f"Publishing to /pose_estimation/image at up to {args.fps} fps"
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Camera read failed, skipping frame")
            return

        t0 = time.time()
        boxes, kp_xy, kp_conf, scores = run_inference(
            self.model, frame, self.score_thresh, self.infer_width
        )
        draw_poses(frame, boxes, kp_xy, kp_conf, scores)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(
            frame, f"{fps:.1f} fps  persons: {len(scores)}", (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
        )

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.pub_image.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-pose → ROS 2 image topic")
    parser.add_argument("--camera",      type=int,   default=0,           help="cv2 camera index")
    parser.add_argument("--width",       type=int,   default=640)
    parser.add_argument("--height",      type=int,   default=480)
    parser.add_argument("--score",       type=float, default=0.5,         help="Detection confidence threshold")
    parser.add_argument("--infer-width", type=int,   default=320,         help="Inference resolution width (multiple of 32)")
    parser.add_argument("--model",       type=str,   default="yolov8n-pose.pt", help="YOLO model file")
    parser.add_argument("--fps",         type=float, default=30.0,        help="Target publish rate in Hz")
    args = parser.parse_args()

    rclpy.init()
    node = PosePublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
