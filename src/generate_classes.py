from pathlib import Path

dataset_root = Path("data/raw/UCF-101")

classes = sorted(
    [
        folder.name
        for folder in dataset_root.iterdir()
        if folder.is_dir()
    ]
)

with open(
    "data/class_names.txt",
    "w"
) as f:

    for cls in classes:

        f.write(
            cls + "\n"
        )

print(
    f"Saved {len(classes)} classes"
)