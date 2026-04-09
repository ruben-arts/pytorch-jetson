"""
Real-time Human Pose Estimation
Displays skeleton keypoints in a local OpenCV window.

Usage:
    python pose_cv2.py [--camera 0] [--width 640] [--height 480]
"""

import argparse
import time
import cv2
import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

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

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
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


def load_model(device: torch.device):
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model


def run_inference(model, frame_bgr, device, score_thresh=0.7, infer_width=320):
    """Run KeypointRCNN on a single BGR frame, return detections."""
    h, w = frame_bgr.shape[:2]
    infer_height = int(h * infer_width / w)
    small = cv2.resize(frame_bgr, (infer_width, infer_height), interpolation=cv2.INTER_LINEAR)

    frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        outputs = model([tensor.to(device)])
    out = outputs[0]

    if device.type == "cuda":
        torch.cuda.empty_cache()

    keep = out["scores"] > score_thresh
    boxes     = out["boxes"][keep].cpu().float().numpy()
    keypoints = out["keypoints"][keep].cpu().float().numpy()  # [N, 17, 3] (x, y, vis)
    scores    = out["scores"][keep].cpu().float().numpy()

    # Scale back to original frame size
    scale_x = w / infer_width
    scale_y = h / infer_height
    if len(keypoints) > 0:
        keypoints[:, :, 0] *= scale_x
        keypoints[:, :, 1] *= scale_y
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    return boxes, keypoints, scores


def draw_poses(frame_bgr, boxes, keypoints, scores):
    """Draw skeleton overlays onto frame in-place."""
    for person_idx, (kps, score) in enumerate(zip(keypoints, scores)):
        # Skeleton lines
        for conn_idx, (a, b) in enumerate(SKELETON):
            if kps[a, 2] > 0.5 and kps[b, 2] > 0.5:
                pt_a = (int(kps[a, 0]), int(kps[a, 1]))
                pt_b = (int(kps[b, 0]), int(kps[b, 1]))
                cv2.line(frame_bgr, pt_a, pt_b, SKELETON_COLORS[conn_idx], 2, cv2.LINE_AA)

        # Keypoint circles
        for i, kp in enumerate(kps):
            if kp[2] > 0.5:
                cv2.circle(frame_bgr, (int(kp[0]), int(kp[1])), 5, KEYPOINT_COLORS[i], -1, cv2.LINE_AA)

        # Confidence label near bounding box
        if person_idx < len(boxes):
            x1, y1 = int(boxes[person_idx][0]), int(boxes[person_idx][1])
            label = f"person {person_idx}  {score:.2f}"
            cv2.putText(frame_bgr, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Pose Estimation → cv2 window")
    parser.add_argument("--camera",      type=int,   default=0,    help="cv2 camera index")
    parser.add_argument("--width",       type=int,   default=640)
    parser.add_argument("--height",      type=int,   default=480)
    parser.add_argument("--score",       type=float, default=0.7,  help="Detection confidence threshold")
    parser.add_argument("--fps-limit",   type=int,   default=30,   help="Max inference FPS")
    parser.add_argument("--infer-width", type=int,   default=320,  help="Resize width before inference (smaller = less memory)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pose] Using device: {device}")

    print("[pose] Loading KeypointRCNN...")
    model = load_model(device)
    print("[pose] Model ready.")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    frame_idx = 0
    t_last    = time.time()
    min_dt    = 1.0 / args.fps_limit

    print("[pose] Streaming — press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[pose] Camera read failed, retrying...")
            time.sleep(0.1)
            continue

        now = time.time()
        if now - t_last < min_dt:
            time.sleep(0.005)
            continue
        t_last = now

        boxes, keypoints, scores = run_inference(model, frame, device, args.score, args.infer_width)
        draw_poses(frame, boxes, keypoints, scores)

        fps = 1.0 / max(time.time() - now, 1e-6)
        cv2.putText(frame, f"{fps:.1f} fps  persons: {len(scores)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
