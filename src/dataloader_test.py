from torch.utils.data import DataLoader

from dataset_v2 import UCF101Dataset


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

videos, labels = next(iter(loader))

print("Videos Shape:", videos.shape)
print("Labels Shape:", labels.shape)
print("Labels:", labels)