import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_segmentation_annotations(image_path, label_path, output_path=None):
    """
    Отрисовывает аннотации сегментации (контуры) для изображения.

    :param image_path: Путь к изображению.
    :param label_path: Путь к файлу аннотаций контуров (сегментации).
    :param output_path: Путь для сохранения изображения с аннотациями (если указан).
    """
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"Файл аннотаций не найден: {label_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return

    h, w, _ = img.shape

    with open(label_path, "r") as f:
        for line in f:
            data = line.strip().split(" ")
            if len(data) < 3:
                continue

            class_id = int(data[0])
            points = list(map(float, data[1:]))
            contour = [(int(x * w), int(y * h)) for x, y in zip(points[::2], points[1::2])]

            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.polylines(img, [np.array(contour, np.int32)], isClosed=True, color=color, thickness=2)

            if contour:
                center_x, center_y = contour[0]
                cv2.putText(img, f"Class {class_id}", (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Изображение с аннотациями сохранено в: {output_path}")


def process_all_annotations(images_dir, labels_dir, output_dir):
    """
    Проходит циклом по изображениям и аннотациям, отрисовывает контуры и сохраняет результат.

    :param images_dir: Путь к папке с изображениями.
    :param labels_dir: Путь к папке с файлами аннотаций.
    :param output_dir: Путь к папке для сохранения обработанных изображений.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = sorted(os.listdir(images_dir))
    labels = sorted(os.listdir(labels_dir))

    for img_file in images:
        if not img_file.endswith(".jpg"):
            continue

        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"

        image_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)
        output_path = os.path.join(output_dir, img_file)

        if os.path.exists(label_path):
            draw_segmentation_annotations(image_path, label_path, output_path)
        else:
            print(f"Не найдена соответствующая аннотация для изображения: {img_file}")


# Пример использования
images_directory = os.path.expanduser("")
labels_directory = os.path.expanduser("")
output_directory = os.path.expanduser("")

process_all_annotations(images_directory, labels_directory, output_directory)
