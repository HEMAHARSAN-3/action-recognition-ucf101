import cv2
import numpy as np
import torch

from model import build_model


# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "outputs/checkpoints/best_model.pth"
VIDEO_PATH = "demo_videos/test_video.avi"
CLASS_FILE = "data/class_names.txt"

NUM_FRAMES = 16
IMAGE_SIZE = 112


# ==========================================
# LOAD CLASS NAMES
# ==========================================

def load_class_names():

    with open(CLASS_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return classes


# ==========================================
# SAMPLE FRAMES
# ==========================================

def sample_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    print(f"\nVideo Path: {video_path}")

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"Total Frames: {total_frames}")

    frames = []

    frame_indices = np.linspace(
        0,
        max(total_frames - 1, 0),
        NUM_FRAMES,
        dtype=int
    )

    for idx in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx
        )

        success, frame = cap.read()

        if success:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

            frames.append(frame)

    cap.release()

    if len(frames) == 0:

        frames = [
            np.zeros(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    3
                ),
                dtype=np.uint8
            )
        ]

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])

    return np.array(frames)


# ==========================================
# LOAD MODEL
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

class_names = load_class_names()

model = build_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()


# ==========================================
# PREPROCESS VIDEO
# ==========================================

frames = sample_frames(VIDEO_PATH)

frames = (
    torch.tensor(
        frames,
        dtype=torch.float32
    ) / 255.0
)

frames = frames.permute(
    3,
    0,
    1,
    2
)

frames = frames.unsqueeze(0)
frames = frames.to(device)


# ==========================================
# INFERENCE
# ==========================================

with torch.no_grad():

    outputs = model(frames)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    top5_probs, top5_indices = torch.topk(
        probabilities,
        k=5,
        dim=1
    )


# ==========================================
# RESULTS
# ==========================================

print("\n" + "=" * 60)
print("TOP-5 ACTION PREDICTIONS")
print("=" * 60)

for i in range(5):

    class_id = top5_indices[0][i].item()

    class_name = class_names[class_id]

    confidence = (
        top5_probs[0][i].item() * 100
    )

    print(
        f"{i+1}. {class_name:<25} "
        f"{confidence:.2f}%"
    )