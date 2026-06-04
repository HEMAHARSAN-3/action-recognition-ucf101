import torch
from torch.utils.data import DataLoader

from dataset_v2 import UCF101Dataset
from model import build_model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    dataset = UCF101Dataset(
        dataset_root="data/raw/UCF-101",
        split_file="data/splits/trainlist01.txt"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    model = build_model()
    model = model.to(device)

    videos, labels = next(iter(loader))

    videos = videos.to(device)

    with torch.no_grad():
        outputs = model(videos)

    print("\nInput Shape:")
    print(videos.shape)

    print("\nOutput Shape:")
    print(outputs.shape)

    print("\nPrediction Tensor:")
    print(outputs[0][:10])


if __name__ == "__main__":
    main()