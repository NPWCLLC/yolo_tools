import os
import shutil
from sklearn.model_selection import train_test_split

images_folder = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/yolo_dataset_boxes_ssl-pups_all/images")
labels_folder = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/yolo_dataset_boxes_ssl-pups_all/labels")

OUTPUT_FOLDER = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/dataset_boxes_75-20-5_ssl-pups_yolo_all")

train_images_folder = os.path.join(OUTPUT_FOLDER, "train", "images")
train_labels_folder = os.path.join(OUTPUT_FOLDER, "train", "labels")

val_images_folder = os.path.join(OUTPUT_FOLDER, "valid", "images")
val_labels_folder = os.path.join(OUTPUT_FOLDER, "valid", "labels")

test_images_folder = os.path.join(OUTPUT_FOLDER, "test", "images")
test_labels_folder = os.path.join(OUTPUT_FOLDER, "test", "labels")

os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)
os.makedirs(test_images_folder, exist_ok=True)
os.makedirs(test_labels_folder, exist_ok=True)

image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('.txt')])

assert len(image_files) == len(label_files), "Число изображений и меток не совпадает!"
assert all(
    os.path.splitext(image_files[i])[0] == os.path.splitext(label_files[i])[0]
    for i in range(len(image_files))
), "Изображения и метки не совпадают по именам!"



# 75-20-5
# Разделение 75%-25% (train и temp)
train_images, temp_images, train_labels, temp_labels = train_test_split(
    image_files, label_files, test_size=0.25, random_state=42  # 25% идут в temp
)

# Разделение temp (25%) на val (20%) и test (5%)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.2, random_state=42  # 20% из temp идут в test
)


def copy_files(file_list, src_folder, dst_folder):
    for file_name in file_list:
        shutil.copy(os.path.join(src_folder, file_name), os.path.join(dst_folder, file_name))

copy_files(train_images, images_folder, train_images_folder)
copy_files(train_labels, labels_folder, train_labels_folder)

copy_files(val_images, images_folder, val_images_folder)
copy_files(val_labels, labels_folder, val_labels_folder)

copy_files(test_images, images_folder, test_images_folder)
copy_files(test_labels, labels_folder, test_labels_folder)


print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")
print(f"Test: {len(test_images)}")
