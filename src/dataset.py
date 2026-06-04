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


if __name__ == "__main__":

    class_file = "data/splits/classInd.txt"
    train_file = "data/splits/trainlist01.txt"

    classes = load_classes(class_file)
    train_samples = load_train_split(train_file)

    print(f"Total Classes: {len(classes)}")
    print(f"Training Videos: {len(train_samples)}")

    print("\nFirst 5 Classes:")
    for i, (name, idx) in enumerate(classes.items()):
        if i == 5:
            break
        print(idx, name)

    print("\nFirst 5 Training Samples:")
    for sample in train_samples[:5]:
        print(sample)