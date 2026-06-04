import cv2
from pathlib import Path


def load_classes(class_file):
    classes = {}

    with open(class_file, "r") as f:
        for line in f:
            idx, class_name = line.strip().split()
            classes[class_name] = int(idx) - 1

    return classes


def load_train_split(split_file):
    samples = []

    with open(split_file, "r") as f:
        for line in f:
            video_path, label = line.strip().split()
            samples.append((video_path, int(label) - 1))

    return samples


def inspect_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Failed to open video")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\nVideo Information")
    print("-" * 30)
    print(f"Frames : {frame_count}")
    print(f"FPS    : {fps}")
    print(f"Size   : {width} x {height}")

    cap.release()


if __name__ == "__main__":

    dataset_root = Path("data/raw/UCF-101")

    train_samples = load_train_split(
        "data/splits/trainlist01.txt"
    )

    first_video = dataset_root / train_samples[0][0]

    print(f"\nVideo Path:\n{first_video}")

    inspect_video(first_video)