"""GPU-accelerated image processing demo on Jetson using raw PyTorch + OpenCV.

Modes (press keys to switch):
  1 - Sobel edge detection (GPU)
  2 - Thermal/heatmap visualization (GPU)
  3 - Live neural cellular automata overlay (GPU)
  q - Quit
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def make_sobel_kernels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create Sobel edge detection kernels."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    # Shape: (out_channels, in_channels, H, W) — apply same kernel to each channel
    kx = kx.unsqueeze(0).unsqueeze(0).to(device)
    ky = ky.unsqueeze(0).unsqueeze(0).to(device)
    return kx, ky


def sobel_edges(frame_t: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edges on GPU. Input: (1, 3, H, W) float [0,1]."""
    gray = frame_t[:, 0:1] * 0.299 + frame_t[:, 1:2] * 0.587 + frame_t[:, 2:3] * 0.114
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2)
    magnitude = magnitude / magnitude.max().clamp(min=1e-5)
    return magnitude.expand(-1, 3, -1, -1)


def thermal_map(frame_t: torch.Tensor) -> torch.Tensor:
    """Convert to thermal/heatmap visualization on GPU."""
    gray = frame_t[:, 0:1] * 0.299 + frame_t[:, 1:2] * 0.587 + frame_t[:, 2:3] * 0.114
    r = torch.clamp(3.0 * gray - 1.0, 0, 1)
    g = torch.where(gray < 0.5, 2.0 * gray, 2.0 * (1.0 - gray))
    b = torch.clamp(1.0 - 3.0 * gray, 0, 1)
    return torch.cat([r, g, b], dim=1)


class NeuralCellularAutomata:
    """Simple neural cellular automata running on GPU.

    Creates organic growing patterns that react to the camera feed.
    """

    def __init__(self, h: int, w: int, device: torch.device) -> None:
        self.device = device
        # State has 16 channels: RGBA visible + 12 hidden
        self.state = torch.zeros(1, 16, h, w, device=device)
        # Seed center
        self.state[:, 3, h // 2, w // 2] = 1.0  # alpha=1 at center

        # Perception kernel: Sobel + identity (3x3) for each channel
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sy = sx.T
        kernel = torch.stack([ident, sx, sy])  # (3, 3, 3)
        # Apply to each channel independently: (48, 1, 3, 3)
        self.kernel = kernel.unsqueeze(1).repeat(16, 1, 1, 1).to(device)

        # Simple update network: two linear layers with relu
        # 48 -> 128 -> 16
        torch.manual_seed(42)
        self.w1 = (torch.randn(128, 48, device=device) * 0.02)
        self.b1 = torch.zeros(128, device=device)
        self.w2 = (torch.randn(16, 128, device=device) * 0.005)
        self.b2 = torch.zeros(16, device=device)

    def step(self, camera_influence: torch.Tensor | None = None) -> torch.Tensor:
        """Run one CA step. Returns RGB image (1, 3, H, W)."""
        # Perceive: depthwise conv
        perceived = F.conv2d(self.state, self.kernel, padding=1, groups=16)  # (1, 48, H, W)

        # Update: pixel-wise MLP
        b, c, h, w = perceived.shape
        x = perceived.permute(0, 2, 3, 1).reshape(-1, c)  # (B*H*W, 48)
        x = F.relu(x @ self.w1.T + self.b1)
        x = x @ self.w2.T + self.b2  # (B*H*W, 16)
        update = x.reshape(b, h, w, 16).permute(0, 3, 1, 2)

        # Stochastic update (only update ~50% of cells per step)
        mask = (torch.rand(1, 1, h, w, device=self.device) > 0.5).float()
        self.state = self.state + update * mask * 0.1

        # Inject camera feed into first 3 channels gently
        if camera_influence is not None:
            self.state[:, :3] = self.state[:, :3] * 0.95 + camera_influence * 0.05

        # Clamp state
        self.state = self.state.clamp(-1, 1)

        # Return visible RGB channels mapped to [0, 1]
        rgb = self.state[:, :3]
        alpha = self.state[:, 3:4].clamp(0, 1)
        return (rgb * 0.5 + 0.5) * alpha + (1 - alpha) * 0.1


def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert BGR frame to (1, 3, H, W) float tensor on device."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_frame(t: torch.Tensor) -> np.ndarray:
    """Convert (1, 3, H, W) tensor to BGR uint8 frame."""
    rgb = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    kx, ky = make_sobel_kernels(device)
    nca = None  # Lazy init after first frame
    mode = 1

    mode_names = {1: "Sobel Edges (GPU)", 2: "Thermal Map (GPU)", 3: "Neural Cellular Automata (GPU)"}
    print("Press 1/2/3 to switch modes, q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_t = frame_to_tensor(frame, device)

        with torch.no_grad():
            if mode == 1:
                result = sobel_edges(frame_t, kx, ky)
            elif mode == 2:
                result = thermal_map(frame_t)
            elif mode == 3:
                if nca is None:
                    _, _, h, w = frame_t.shape
                    nca = NeuralCellularAutomata(h, w, device)
                # Run a few steps per frame for visible evolution
                for _ in range(4):
                    result = nca.step(camera_influence=frame_t)

        output = tensor_to_frame(result)

        # HUD
        cv2.putText(output, mode_names[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, "Press 1/2/3 to switch, q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("PyTorch GPU Demo", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            mode = key - ord("0")
            if mode == 3 and nca is not None:
                # Reset NCA when switching to it
                _, _, h, w = frame_t.shape
                nca = NeuralCellularAutomata(h, w, device)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
