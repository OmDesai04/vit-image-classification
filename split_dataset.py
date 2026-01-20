import os
import random
import shutil


SOURCE_DIR = "dataset"
DEST_DIR = "split_dataset"
TRAIN_RATIO = 0.5
TEST_RATIO = 0.2
VAL_RATIO = 0.2
SEED = 42


random.seed(SEED)

for split in ["train", "test", "val", "unused"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total_images = len(images)

    train_count = int(total_images * TRAIN_RATIO)
    test_count = int(total_images * TEST_RATIO)
    val_count = int(total_images * VAL_RATIO)

    train_images = images[:train_count]
    test_images = images[train_count:train_count + test_count]
    val_images = images[train_count + test_count:
                        train_count + test_count + val_count]
    unused_images = images[train_count + test_count + val_count:]

    splits = {
        "train": train_images,
        "test": test_images,
        "val": val_images,
        "unused": unused_images
    }

    for split_name, split_images in splits.items():
        split_class_dir = os.path.join(DEST_DIR, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy(src, dst)

    print(
        f"{class_name}: total={total_images}, "
        f"train={len(train_images)}, "
        f"test={len(test_images)}, "
        f"val={len(val_images)}, "
        f"unused={len(unused_images)}"
    )

print("\nDataset split completed.")
