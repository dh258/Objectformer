import shutil
from pathlib import Path
import random
from typing import List, Tuple
import cv2
import numpy as np


def find_mask_for_image(image_name: str, mask_dir: Path) -> str:
    """Find corresponding mask file for a tampered image"""
    # Remove extension and try to match with available masks
    base_name = Path(image_name).stem

    # Direct match first
    for mask_file in mask_dir.glob(f"{base_name}.*"):
        return mask_file.name

    # Try variations (cpmv, splc suffixes)
    for mask_file in mask_dir.iterdir():
        if mask_file.stem.startswith(base_name):
            return mask_file.name

    return None


def convert_dataset(
    source_dir: str = "training_dataset/ktp-only",
    target_dir: str = "training_dataset",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Convert sorted_manual dataset to ObjectFormer format

    Args:
        source_dir: Path to sorted_manual directory
        target_dir: Path to target dataset directory
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split
        test_ratio: Ratio for test split
    """

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory structure
    images_dir = target_path / "images"
    masks_dir = target_path / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    print(f"Converting dataset from {source_dir} to {target_dir}")

    # Step 1: Copy genuine images (label=0) and create blank masks
    genuine_dir = source_path / "genuine"
    genuine_files = []

    for img_file in genuine_dir.iterdir():
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            # Copy to images directory
            target_img = images_dir / img_file.name
            shutil.copy2(img_file, target_img)

            # Create blank mask for genuine image
            mask_name = f"{img_file.stem}_mask.png"
            target_mask = masks_dir / mask_name

            # Read image to get dimensions for blank mask
            img = cv2.imread(str(img_file))
            height, width = img.shape[:2]

            # Create blank mask (all zeros = authentic pixels)
            blank_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.imwrite(str(target_mask), blank_mask)

            # Record for annotation with proper mask path
            genuine_files.append((img_file.name, mask_name, 0))
            print(f"Copied genuine: {img_file.name} -> blank mask: {mask_name}")

    # Step 2: Copy tampered images and masks (label=1)
    tampered_dir = source_path / "tampered"
    mask_source_dir = source_path / "masks"
    tampered_files = []

    for img_file in tampered_dir.iterdir():
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            # Copy image to images directory
            target_img = images_dir / img_file.name
            shutil.copy2(img_file, target_img)

            # Find corresponding mask
            mask_name = find_mask_for_image(img_file.name, mask_source_dir)
            if mask_name:
                # Copy mask to masks directory
                mask_source = mask_source_dir / mask_name
                target_mask = masks_dir / mask_name
                shutil.copy2(mask_source, target_mask)

                # Record for annotation
                tampered_files.append((img_file.name, mask_name, 1))
                print(f"Copied tampered: {img_file.name} -> mask: {mask_name}")
            else:
                print(f"WARNING: No mask found for {img_file.name}")
                # Still add to tampered but without mask (will need manual check)
                tampered_files.append((img_file.name, "", 1))

    # Step 3: Create stratified train/val/test splits
    # Shuffle each class separately to ensure balanced distribution
    random.shuffle(genuine_files)
    random.shuffle(tampered_files)
    
    # Calculate splits for each class separately
    genuine_count = len(genuine_files)
    tampered_count = len(tampered_files)
    
    # Split genuine files
    genuine_train_end = int(genuine_count * train_ratio)
    genuine_val_end = genuine_train_end + int(genuine_count * val_ratio)
    
    genuine_train = genuine_files[:genuine_train_end]
    genuine_val = genuine_files[genuine_train_end:genuine_val_end]
    genuine_test = genuine_files[genuine_val_end:]
    
    # Split tampered files
    tampered_train_end = int(tampered_count * train_ratio)
    tampered_val_end = tampered_train_end + int(tampered_count * val_ratio)
    
    tampered_train = tampered_files[:tampered_train_end]
    tampered_val = tampered_files[tampered_train_end:tampered_val_end]
    tampered_test = tampered_files[tampered_val_end:]
    
    # Combine and shuffle each split
    train_files = genuine_train + tampered_train
    val_files = genuine_val + tampered_val
    test_files = genuine_test + tampered_test
    
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    # Step 4: Write annotation files
    def write_split_file(filename: str, file_list: List[Tuple[str, str, int]]):
        with open(target_path / filename, "w") as f:
            for img_name, mask_name, label in file_list:
                # All entries now have proper mask paths (blank for genuine, real for tampered)
                mask_path = f"masks/{mask_name}" if mask_name else ""
                f.write(f"images/{img_name}\t{mask_path}\t{label}\n")

    write_split_file("train_split.txt", train_files)
    write_split_file("val_split.txt", val_files)
    write_split_file("test_split.txt", test_files)

    # Step 5: Print summary with class distribution
    all_files = train_files + val_files + test_files
    
    def count_classes(file_list):
        genuine = sum(1 for _, _, label in file_list if label == 0)
        tampered = sum(1 for _, _, label in file_list if label == 1)
        return genuine, tampered
    
    train_genuine, train_tampered = count_classes(train_files)
    val_genuine, val_tampered = count_classes(val_files)
    test_genuine, test_tampered = count_classes(test_files)
    
    print("\n" + "=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    print(f"Total genuine images: {len(genuine_files)}")
    print(f"Total tampered images: {len(tampered_files)}")
    print(f"Total files: {len(all_files)}")
    print(f"\nStratified split distribution:")
    print(f"  Train: {len(train_files)} files ({len(train_files) / len(all_files) * 100:.1f}%) - {train_genuine} genuine, {train_tampered} tampered")
    print(f"  Val:   {len(val_files)} files ({len(val_files) / len(all_files) * 100:.1f}%) - {val_genuine} genuine, {val_tampered} tampered")
    print(f"  Test:  {len(test_files)} files ({len(test_files) / len(all_files) * 100:.1f}%) - {test_genuine} genuine, {test_tampered} tampered")

    print(f"\nFiles created:")
    print(f"  Images directory: {images_dir} ({len(list(images_dir.iterdir()))} files)")
    print(f"  Masks directory: {masks_dir} ({len(list(masks_dir.iterdir()))} files)")
    print(f"  train_split.txt: {len(train_files)} entries")
    print(f"  val_split.txt: {len(val_files)} entries")
    print(f"  test_split.txt: {len(test_files)} entries")

    # Check for missing masks
    missing_masks = [f for f in tampered_files if not f[1]]
    if missing_masks:
        print(f"\nWARNING: {len(missing_masks)} tampered images without masks:")
        for img_name, _, _ in missing_masks:
            print(f"  - {img_name}")


if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)

    convert_dataset()

    print("\nDataset conversion complete!")
    print("\nNext steps:")
    print("1. Update your config file ROOT_DIR to point to 'training_dataset'")
    print("2. Set TRAIN_SPLIT: 'train_split.txt', VAL_SPLIT: 'val_split.txt', etc.")
    print("3. Run a test training to verify everything works")
