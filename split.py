import os
import shutil
import random

# Paths
train_dir = r"C:\Project\part2\archive\asl_alphabet_train\asl_alphabet_train"
test_dir = r"C:\Project\part2\archive\asl_alphabet_test"

# Percentage of images to move
split_ratio = 0.2  # 20%

# Create test directory if not exists
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Loop through each class folder
for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)

    if os.path.isdir(folder_path):

        images = os.listdir(folder_path)
        random.shuffle(images)

        move_count = int(len(images) * split_ratio)

        # Create same folder in test directory
        test_class_path = os.path.join(test_dir, folder)
        if not os.path.exists(test_class_path):
            os.makedirs(test_class_path)

        # Move images
        for img in images[:move_count]:
            src = os.path.join(folder_path, img)
            dst = os.path.join(test_class_path, img)

            shutil.move(src, dst)

        print(f"✅ Moved {move_count} images from {folder} to test folder")

print("\n🎉 Dataset split completed successfully!")