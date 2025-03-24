import os
import shutil
from sklearn.model_selection import train_test_split

images_folder = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/yolo_dataset_boxes_ssl-pups_all/images")
labels_folder = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/yolo_dataset_boxes_ssl-pups_all/labels")

OUTPUT_FOLDER = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/dataset_boxes_75-20-5_ssl-pups_yolo_bg_15")

train_images_folder = os.path.join(OUTPUT_FOLDER, "train", "images")
train_labels_folder = os.path.join(OUTPUT_FOLDER, "train", "labels")

val_images_folder = os.path.join(OUTPUT_FOLDER, "valid", "images")
val_labels_folder = os.path.join(OUTPUT_FOLDER, "valid", "labels")

test_images_folder = os.path.join(OUTPUT_FOLDER, "test", "images")
test_labels_folder = os.path.join(OUTPUT_FOLDER, "test", "labels")

# Создание папок
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

# Фильтрация пустых меток
# Оставляем только непустые метки
valid_images = []
valid_labels = []
empty_images = []
empty_labels = []

for img, lbl in zip(image_files, label_files):
    label_path = os.path.join(labels_folder, lbl)
    with open(label_path, 'r') as f:
        content = f.read().strip()
    if content:  # Если файл не пустой
        valid_images.append(img)
        valid_labels.append(lbl)
    else:  # Если файл пустой
        empty_images.append(img)
        empty_labels.append(lbl)

print(f"Непустых аннотаций: {len(valid_labels)}")
print(f"Пустых аннотаций: {len(empty_labels)}")

# 75-20-5
# Разделение 75%-25% (train и temp)
train_images, temp_images, train_labels, temp_labels = train_test_split(
    valid_images, valid_labels, test_size=0.25, random_state=42  # 25% идут в temp
)

# Разделение temp (25%) на val (20%) и test (5%)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.2, random_state=42  # 20% из temp идут в test
)


# Копирование файлов
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

# Добавление гибкого процента пустых аннотаций в папку train
empty_percentage = 15  # Укажите процент пустых аннотаций для добавления (10-50%)
num_empty_to_add = len(train_labels) * empty_percentage // 100

print(f"Добавляем {num_empty_to_add} пустых аннотаций в Train ({empty_percentage}% от размера обучающего набора)")

# Проверим, что хватает пустых аннотаций
if num_empty_to_add > len(empty_images):
    print("Предупреждение: не хватает доступных пустых аннотаций. Используем максимум доступных.")
    num_empty_to_add = len(empty_images)

# Выбираем нужное количество пустых файлов
empty_images_to_add = empty_images[:num_empty_to_add]
empty_labels_to_add = empty_labels[:num_empty_to_add]

# Копируем пустые файлы в train
copy_files(empty_images_to_add, images_folder, train_images_folder)
copy_files(empty_labels_to_add, labels_folder, train_labels_folder)

print(f"Обучающий набор теперь содержит {len(os.listdir(train_images_folder))} изображений.")
