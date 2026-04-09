"""
Real-time Human Pose Estimation
Streams skeleton keypoints to a remote Rerun viewer over the network.

Usage:
    python pose.py --rerun-addr <remote-ip>:9876
"""

import argparse
import time
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import rerun as rr
import rerun.blueprint as rrb

# COCO keypoint skeleton connections
SKELETON = [
    (0, 1), (0, 2),             # nose -> eyes
    (1, 3), (2, 4),             # eyes -> ears
    (5, 6),                     # shoulders
    (5, 7), (7, 9),             # left arm
    (6, 8), (8, 10),            # right arm
    (5, 11), (6, 12),           # shoulders -> hips
    (11, 12),                   # hips
    (11, 13), (13, 15),         # left leg
    (12, 14), (14, 16),         # right leg
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Color palette per body part (RGB)
PART_COLORS = {
    "head":  (255, 200,  50),
    "arm":   ( 50, 200, 255),
    "torso": ( 80, 255, 120),
    "leg":   (255,  80, 150),
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
    if device.type == "cuda":
        model.half()
    return model


def run_inference(model, frame_bgr, device, score_thresh=0.7, infer_width=320):
    """Run KeypointRCNN on a single BGR frame, return detections."""
    h, w = frame_bgr.shape[:2]
    infer_height = int(h * infer_width / w)
    small = cv2.resize(frame_bgr, (infer_width, infer_height), interpolation=cv2.INTER_LINEAR)

    frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    if device.type == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        outputs = model([tensor.to(device)])
    out = outputs[0]

    if device.type == "cuda":
        torch.cuda.empty_cache()

    keep = out["scores"] > score_thresh
    boxes     = out["boxes"][keep].cpu().float().numpy()
    keypoints = out["keypoints"][keep].cpu().float().numpy()   # [N, 17, 3] (x, y, vis)
    scores    = out["scores"][keep].cpu().float().numpy()

    # Scale keypoints and boxes back to original frame size
    scale_x = w / infer_width
    scale_y = h / infer_height
    if len(keypoints) > 0:
        keypoints[:, :, 0] *= scale_x
        keypoints[:, :, 1] *= scale_y
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    return boxes, keypoints, scores


def log_frame(frame_bgr, boxes, keypoints, scores, frame_idx: int, log_image_every: int = 3):
    """Log everything to Rerun for this frame."""
    rr.set_time("frame", sequence=frame_idx)

    # --- Raw camera image (throttled to reduce bandwidth/memory pressure) ---
    if frame_idx % log_image_every == 0:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rr.log("camera/image", rr.Image(frame_rgb))

    # --- Per-person skeleton ---
    for person_idx, (kps, score) in enumerate(zip(keypoints, scores)):
        entity = f"camera/pose/person_{person_idx}"

        # Keypoints (visible ones only)
        visible = kps[:, 2] > 0.5
        pts  = kps[visible, :2]
        cols = [KEYPOINT_COLORS[i] for i, v in enumerate(visible) if v]
        labs = [KEYPOINT_NAMES[i]  for i, v in enumerate(visible) if v]

        if len(pts) > 0:
            rr.log(
                f"{entity}/keypoints",
                rr.Points2D(
                    pts,
                    radii=5.0,
                    colors=cols,
                    labels=labs,
                ),
            )

        # Skeleton lines
        line_starts, line_ends, line_colors = [], [], []
        for conn_idx, (a, b) in enumerate(SKELETON):
            if kps[a, 2] > 0.5 and kps[b, 2] > 0.5:
                line_starts.append(kps[a, :2])
                line_ends.append(kps[b, :2])
                line_colors.append(SKELETON_COLORS[conn_idx])

        if line_starts:
            lines = [[s, e] for s, e in zip(line_starts, line_ends)]
            rr.log(
                f"{entity}/skeleton",
                rr.LineStrips2D(
                    lines,
                    radii=2.5,
                    colors=line_colors,
                ),
            )

        # Confidence label near bounding box
        if len(boxes) > person_idx:
            x1, y1 = boxes[person_idx][:2]
            rr.log(
                f"{entity}/label",
                rr.TextLog(f"person {person_idx}  {score:.2f}"),
            )

    # --- Stats scalar ---
    rr.log("stats/num_persons", rr.Scalars(len(keypoints)))


def setup_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Camera + Pose", contents=["camera/**"]),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Person count", contents=["stats/**"]),
                rrb.TextLogView(name="Labels", contents=["camera/pose/**/label"]),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Pose Estimation → Rerun")
    parser.add_argument("--camera",     type=int,   default=0,          help="cv2 camera index")
    parser.add_argument("--rerun-addr", type=str,   default=None,       help="Remote Rerun addr, e.g. 192.168.1.42:9876")
    parser.add_argument("--width",      type=int,   default=640)
    parser.add_argument("--height",     type=int,   default=480)
    parser.add_argument("--score",      type=float, default=0.7,        help="Detection confidence threshold")
    parser.add_argument("--fps-limit",  type=int,   default=30,         help="Max inference FPS")
    parser.add_argument("--infer-width", type=int,  default=320,        help="Resize width before inference (smaller = less memory)")
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pose] Using device: {device}")

    # --- Rerun ---
    rr.init("pose_estimation", spawn=False)
    if args.rerun_addr:
        if args.rerun_addr == "true":
            rr.connect_grpc()
        else:
            rr.connect_grpc(args.rerun_addr)
        print(f"[rerun] Streaming to {args.rerun_addr}")
    else:
        rr.spawn()
        print("[rerun] Spawned local viewer")

    rr.send_blueprint(setup_blueprint())

    # --- Model ---
    print("[pose] Loading KeypointRCNN...")
    model = load_model(device)
    print("[pose] Model ready.")

    # --- Camera ---
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    frame_idx   = 0
    t_last      = time.time()
    min_dt      = 1.0 / args.fps_limit

    print("[pose] Streaming — press Ctrl+C to stop.")
    try:
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

            boxes, keypoints, scores = run_inference(
                model, frame, device, args.score, args.infer_width
            )
            log_frame(frame, boxes, keypoints, scores, frame_idx)

            fps = 1.0 / max(time.time() - now, 1e-6)
            print(f"\r[pose] frame {frame_idx:5d} | persons {len(scores):2d} | {fps:.1f} fps", end="")
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[pose] Stopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()