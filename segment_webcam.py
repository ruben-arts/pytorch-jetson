"""Simple webcam segmentation using torchvision's DeepLabV3 on Jetson."""

import cv2
import numpy as np
import torch
from torchvision import models, transforms

# Colors for the 21 PASCAL VOC classes
COLORS = np.random.RandomState(42).randint(0, 255, (21, 3), dtype=np.uint8)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained DeepLabV3 with MobileNetV3 backbone (lightweight)
    print("Loading model...")
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    )
    model = model.to(device).eval()
    print("Model loaded!")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)["out"]
            pred = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

        # Color the segmentation mask
        mask = COLORS[pred]

        # Blend original frame with segmentation overlay
        blended = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)

        # Show detected class labels
        detected = [CLASSES[i] for i in np.unique(pred) if i != 0]
        if detected:
            label = "Detected: " + ", ".join(detected)
            cv2.putText(blended, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Segmentation", blended)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
