"""
Вспомогательные функции
"""

import os
import glob
import logging
import zipfile

from pathlib import Path

LOG_FOLDER = 'logs'
IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')
VIDEO_FORMATS = ('.mp4', '.avi', '.mov')
SUBSET_NAMES = ('train', 'val', 'test', 'valid')

Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)


def setting_logs(log_file)-> logging:
    # Настройка логирования
    log_file = os.path.join(LOG_FOLDER, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # Логи в файл
            logging.StreamHandler()  # Логи в консоль
        ]
    )
    return logging

def count_annotations(dataset_dir):
    """
    Функция для подсчета аннотаций для каждого класса в наборе данных,
    а также для подсчета пустых аннотаций.
    :param dataset_dir: Путь к папке набора данных.
    :return: Список изображений, словарь с количеством аннотаций по классам,
             общее количество аннотаций, количество пустых аннотаций и их доля.
    """
    class_counts = {}
    total_annotations = 0
    empty_annotations = 0  # Для подсчета пустых файлов аннотаций
    image_files = [f for f in os.listdir(os.path.join(dataset_dir, "images")) if f.endswith(IMAGE_FORMATS)]

    for image_file in image_files:
        annotation_file = f"{Path(image_file).stem}.txt"
        annotation_path = os.path.join(dataset_dir, "labels", annotation_file)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
                if not lines:  # Файл аннотации пустой
                    empty_annotations += 1
                else:
                    total_annotations += len(lines)
                    for line in lines:
                        class_id = int(line.split()[0])  # Первый элемент — это ID класса
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                        else:
                            class_counts[class_id] = 1

    total_annotations += empty_annotations
    # Рассчитываем долю пустых аннотаций
    empty_annotation_ratio = empty_annotations / total_annotations if total_annotations > 0 and empty_annotations > 0  else 0

    return image_files, class_counts, total_annotations, empty_annotations, empty_annotation_ratio


def find_all_image_files(directory):
    """
    Возвращает список путей к изображениям в указанной директории и её поддиректориях,
    соответствующим расширениям из SUFFIXES.

    :param directory: Директория для поиска изображений.
    :return: Список путей к файлам изображений.
    """

    image_paths = []
    for suffix in IMAGE_FORMATS:
        image_paths.extend(glob.glob(os.path.join(directory, f"**/*{suffix}"), recursive=True))

    return image_paths

def find_all_video_files(directory):
    """
    Рекурсивный поиск всех видео файлов поддерживаемым форматом в заданной директории и её поддиректориях.
    """
    video_files = []  # Список для хранения найденных файлов

    # Рекурсивный обход всех папок
    for root, _, files in os.walk(directory):
        for file in files:
            # Проверяем, является ли файл поддерживаемым форматом
            if Path(file).suffix.lower() in VIDEO_FORMATS:
                video_files.append(os.path.join(root, file))  # Сохраняем полный путь к файлу

    return video_files

def extract_zip_file(zip_path, current_logging):
    """
    Распаковывает zip-архив в директории архива.

    :param current_logging:
    :param zip_path: Путь к zip-файлу.
    """

    try:

        # Проверяем, существует ли zip-файл
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Файл {zip_path} не найден!")

        extract_to = Path(zip_path).parent

        # Открываем zip-файл и извлекаем его содержимое
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            current_logging.info(f"Архив {zip_path} успешно распакован в {extract_to}.")

    except zipfile.BadZipFile:
        current_logging.error(f"Ошибка: файл {zip_path} не корректный zip-архив.")
    except Exception as e:
        current_logging.error(f"Произошла ошибка: {e}")


def print_statistics(dataset_dir, current_logging):
    """
    Функция для вывода статистики по набору данных, включая расчет доли:
    1. Непустых аннотаций в val относительно train.
    2. Изображений в val относительно train.

    :param dataset_dir: Путь к папке набора данных.
    :param current_logging: Логгер для записи выводов.
    """
    dataset_folder = Path(dataset_dir).name
    current_logging.info(f"Папка датасета: {dataset_folder}")

    # Для хранения статистики по train и val
    empty_annotations_train = 0
    train_non_empty_annotations = 0
    val_non_empty_annotations = 0
    train_images_count = 0
    val_images_count = 0

    for subset in SUBSET_NAMES:
        subset_path = os.path.join(dataset_dir, subset)
        if os.path.exists(subset_path):
            image_files, class_counts, total_annotations, empty_annotations, empty_annotation_ratio = count_annotations(
                subset_path)

            # Считаем количество изображений
            images_count = len(image_files)

            # Обновляем переменные в зависимости от подмножества
            if subset == "train":  # Для train
                train_non_empty_annotations = total_annotations  # Общее число непустых аннотаций
                train_images_count = images_count  # Число изображений
                empty_annotations_train = empty_annotations
            elif subset in ("val", "valid"):  # Для val
                val_non_empty_annotations = total_annotations  # Непустые аннотации
                val_images_count = images_count  # Число изображений

            current_logging.info(f"Статистика для набора {subset}:")
            current_logging.info(f"Количество изображений: {images_count}")
            current_logging.info(f"Общее количество аннотаций: {total_annotations}")
            current_logging.info(f"Число пустых аннотаций: {empty_annotations} ({empty_annotation_ratio:.2%})")
            current_logging.info("Аннотации по классам:")

            # Подсчитываем количество аннотаций по каждому классу
            for class_id, count in class_counts.items():
                current_logging.info(f"Класс {class_id}: {count} аннотаций")

    # Расчет доли непустых аннотаций в val относительно train
    if train_non_empty_annotations > 0 and val_images_count > 0:
        val_to_train_annotation_ratio = val_non_empty_annotations / train_non_empty_annotations
        current_logging.info(f"Доля непустых аннотаций в val относительно train: {val_to_train_annotation_ratio:.2%}")


    # Расчет доли изображений в val относительно train
    if train_images_count > 0 and val_images_count > 0:
        val_to_train_image_ratio = val_images_count / train_images_count
        current_logging.info(f"Доля изображений в val относительно train: {val_to_train_image_ratio:.2%}")


    current_logging.info("----------------------------")

