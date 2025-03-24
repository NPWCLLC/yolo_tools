import os
import numpy as np


def convert_segmentation_to_detection(input_dir, output_dir):
    """
    Конвертация файлов сегментации YOLO в файл ограничивающих боксов.

    :param input_dir: Путь к папке с YOLO сегментами (labels).
    :param output_dir: Папка для сохранения YOLO боксов.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".txt"):
            continue

        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)

        with open(input_file_path, 'r') as file:
            lines = file.readlines()

        converted_boxes = []

        for line in lines:
            elements = line.strip().split()
            class_id = elements[0]
            polygon_points = np.array(elements[1:], dtype=float).reshape(-1, 2)

            # Вычисляем ограничивающую рамку (в нормализованных координатах)
            x_min, y_min = np.min(polygon_points, axis=0)
            x_max, y_max = np.max(polygon_points, axis=0)

            # Центр и размеры бокса (нормализация уже выполнена)
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Добавляем строку в YOLO формате
            converted_boxes.append(f"{class_id} {x_center:.10f} {y_center:.10f} {bbox_width:.10f} {bbox_height:.10f}")

        # Сохраняем новый YOLO файл
        with open(output_file_path, 'w') as output_file:
            output_file.write("\n".join(converted_boxes))


def process_yolo_dataset(root_dir):
    """
    Форматирует папку YOLO со структурами train/valid/test.

    :param root_dir: Директория с YOLO датасетом (включая train, valid, test).
    """
    subsets = ['train', 'valid', 'test']

    for subset in subsets:
        subset_path = os.path.join(root_dir, subset, 'labels')
        output_path = os.path.join(root_dir, f"{subset}_converted", 'labels')

        if not os.path.exists(subset_path):
            print(f"Пропущена папка: {subset_path}")
            continue


        convert_segmentation_to_detection(subset_path, output_path)

if __name__ == "__main__":
    root_yolo_dir = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/SSL-CSL-Segm.v2i.yolov11_balance-class_weight")  # Папка с train, valid, test

    process_yolo_dataset(root_yolo_dir)

