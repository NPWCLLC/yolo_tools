import os
from ultralytics.data.utils import visualize_image_annotations
# ONLY FOR BOXES
def for_boxes():
    # Использование функции
    image_path = os.path.expanduser("~/TRAIN_DATA/SSL-CSL-SEG/dataset_ssl-csl_yolo_segm_t80-v20_bg_t50_v10/train/images/0-0_20220822_160024_02-22_jpg.rf.89d8c319fa685d0c167b55af2a696fe3_aug_0.jpg") # Путь к изображению
    label_path = os.path.expanduser("~/TRAIN_DATA/SSL-CSL-SEG/dataset_ssl-csl_yolo_segm_t80-v20_bg_t50_v10/train/labels/0-0_20220822_160024_02-22_jpg.rf.89d8c319fa685d0c167b55af2a696fe3_aug_0.txt") # Путь к аннотациям YOLO

    label_map = {  # Define the label map with all annotated class labels.
        0: "csl",
        1: "ssl",
    }

    # Visualize
    visualize_image_annotations(
        image_path=image_path,
        txt_path=label_path,
        label_map=label_map,
    )

import os
import cv2
import numpy as np
# Показываем изображение с аннотациями
import matplotlib.pyplot as plt


def draw_segmentation_annotations(image_path, label_path, output_path=None):
    """
    Отрисовывает аннотации сегментации (контуры) для изображения.

    :param image_path: Путь к изображению.
    :param label_path: Путь к файлу аннотаций контуров (сегментации).
    :param output_path: Путь для сохранения изображения с аннотациями (если указан).
    """
    # Проверяем, что файл изображения и меток существует
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"Файл аннотаций не найден: {label_path}")
        return

    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return

    # Получаем размеры изображения
    h, w, _ = img.shape

    # Читаем аннотации из файла
    with open(label_path, "r") as f:
        for line in f:
            data = line.strip().split(" ")
            if len(data) < 3:
                continue

            # Первый элемент — ID класса, остальные — набор координат точек (x1, y1, x2, y2, ...)
            class_id = int(data[0])
            points = list(map(float, data[1:]))

            # Преобразуем нормализованные координаты (0-1) в пиксели
            contour = [(int(x * w), int(y * h)) for x, y in zip(points[::2], points[1::2])]

            # Отрисовываем контур (используем случайный цвет для каждого класса)
            color = tuple(np.random.randint(0, 255, size=3).tolist())  # Случайный цвет
            cv2.polylines(img, [np.array(contour, np.int32)], isClosed=True, color=color, thickness=2)

            # Отмечаем ID класса в центре первого сегмента
            if contour:
                center_x, center_y = contour[0]
                cv2.putText(img, f"Class {class_id}", (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



    # Преобразуем изображение из BGR в RGB для matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

    # Сохраняем изображение с аннотациями, если указан путь
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Изображение с аннотациями сохранено в: {output_path}")


# Пример использования
image_path = os.path.expanduser("") # Путь к изображению
label_path = os.path.expanduser("") # Путь к аннотациям YOLO


draw_segmentation_annotations(image_path, label_path)
