import re
import shutil
from pathlib import Path

import cv2
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

DEBUG = False
SCALE = 1.3
IS_CONTOURS = False


def remove_special_characters(input_string):
    return re.sub(r'[^a-zA-Z0-9_]', '_', input_string)


images_folder = os.path.expanduser("~/DENIS_GAEV/TRAIN/AUC_data/SSL_Pup/SmalOriginal/Image")
masks_folder = os.path.expanduser("~/DENIS_GAEV/TRAIN/AUC_data/SSL_Pup/SmalOriginal/Mask")

OUTPUT_DIR = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV")
OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, "yolo_dataset_{0}".format("segmentation" if IS_CONTOURS else "boxes"))

output_images_folder = os.path.join(OUTPUT_FOLDER, "images")
output_labels_folder = os.path.join(OUTPUT_FOLDER, "labels")

os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

errors = []
total_annotations = 0
total_files = 0
files = os.listdir(images_folder)

for i, image_name in enumerate(tqdm(files, total=len(files))):

    if not image_name.endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(images_folder, image_name)
    mask_path = os.path.join(masks_folder, image_name.replace(".jpg", ".png").replace(".jpeg", ".png"))

    if not os.path.isfile(mask_path):
        errors.append(f"Пропущено: Маска для изображения {image_name} не найдена.")
        continue

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        errors.append(f"Ошибка загрузки {image_name} или его маски.")
        continue

    # Преобразуем BGR изображение в RGB для отображения
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Маска должна быть либо бинарной, либо применяться с прозрачностью
    masked_image = image_rgb.copy()
    masked_image[mask == 0] = [0, 0, 0]  # Наносим черный цвет на области маски с нулевым значением

    if DEBUG:
        # Преобразуем изображение из BGR в RGB для правильного отображения через matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_normalized = mask / 255.0
        color = np.array([1, 0, 0])

        overlay = image_rgb * (1 - mask_normalized[..., None]) + color * mask_normalized[..., None] * 255

        overlay = overlay.astype(np.uint8)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(image_rgb)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlay with Transparency")
        plt.imshow(overlay)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    img_height, img_width = image.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_annotations = []
    # if not contours:
    #     errors.append(f"Пропущено: Маска для изображения {image_name} не найден контур.")
    #     continue


    for contour in contours:
        # Найти центр контура: моменты изображения
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Убедиться, что контур не пустой
            cx = int(M["m10"] / M["m00"])  # Центр X
            cy = int(M["m01"] / M["m00"])  # Центр Y
        else:
            continue  # Пропускать контуры без площади

        # Масштабируем контур или bounding box
        if IS_CONTOURS:
            normalized_contour = []
            debug_contours = []
            for point in contour:
                x, y = point[0]
                # Увеличение расстояния от центра с учётом коэффициента
                new_x = cx + SCALE * (x - cx)
                new_y = cy + SCALE * (y - cy)

                # Ограничиваем в пределах изображения
                new_x = max(0, min(img_width - 1, new_x))
                new_y = max(0, min(img_height - 1, new_y))

                # Нормализуем координаты
                norm_x = new_x / img_width
                norm_y = new_y / img_height
                normalized_contour.append(f"{norm_x:.10f} {norm_y:.10f}")
                debug_contours.append([new_x, new_y])

            if DEBUG:
                scaled_contour = np.array(debug_contours, dtype=np.int32)
                # Рисуем контур на изображении
                cv2.polylines(image, [scaled_contour], isClosed=True, color=(0, 255, 0), thickness=2)

            # Формируем аннотацию для контуров
            annotation = f"0 " + " ".join(normalized_contour)

        else:
            # Определение min, max координат контура
            x_min = min(point[0][0] for point in contour)
            y_min = min(point[0][1] for point in contour)
            x_max = max(point[0][0] for point in contour)
            y_max = max(point[0][1] for point in contour)

            # Масштабируем bounding box (x_min, y_min, x_max, y_max)
            x_min_scaled = cx + SCALE * (x_min - cx)
            y_min_scaled = cy + SCALE * (y_min - cy)
            x_max_scaled = cx + SCALE * (x_max - cx)
            y_max_scaled = cy + SCALE * (y_max - cy)

            # Нормализуем координаты для YOLO
            x_min_norm = max(0, x_min_scaled / img_width)
            y_min_norm = max(0, y_min_scaled / img_height)
            x_max_norm = min(1, x_max_scaled / img_width)
            y_max_norm = min(1, y_max_scaled / img_height)

            # Пересчитываем центр и размеры бокса
            cx_norm = (x_min_norm + x_max_norm) / 2
            cy_norm = (y_min_norm + y_max_norm) / 2
            width_norm = x_max_norm - x_min_norm
            height_norm = y_max_norm - y_min_norm

            # Проверяем корректность размеров
            if width_norm <= 0 or height_norm <= 0:
                continue

            if DEBUG:
                # Рисуем прямоугольник на изображении
                cv2.rectangle(image,
                              (int(x_min_scaled), int(y_min_scaled)),
                              (int(x_max_scaled), int(y_max_scaled)),
                              color=(0, 0, 255),
                              thickness=2)

            # Формируем аннотацию для bounding box
            annotation = f"0 {cx_norm:.10f} {cy_norm:.10f} {width_norm:.10f} {height_norm:.10f}"

        if DEBUG:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 7))
            plt.imshow(image_rgb)
            plt.axis("off")
            plt.title("Contours and Bounding Boxes")
            plt.show()

        yolo_annotations.append(annotation)

    new_name = remove_special_characters(Path(image_path).stem)
    yolo_file_path = os.path.join(output_labels_folder, f"{new_name}_{i}.txt")

    with open(yolo_file_path, "w") as yolo_file:
        yolo_file.write("\n".join(yolo_annotations))

    shutil.copy(image_path, os.path.join(output_images_folder, f"{new_name}_{i}{Path(image_path).suffix}"))

    total_annotations += len(yolo_annotations)
    total_files += 1

print(f"Total annotations: {total_annotations}")
print(f"Total files: {total_files}")
print(f"Errors no append: {len(errors)}")

with open(os.path.join(OUTPUT_DIR, "error_no_append_convert_data.txt"), "w", encoding="utf-8") as file:
    for line in errors:
        file.write(line + "\n")
