import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from utils import setting_logs, print_statistics

LOG_FILE = 'split_dataset.log'
logging = setting_logs(LOG_FILE)

class DatasetManager:
    def __init__(self, images_path, labels_path, output_folder):
        self.images_path = os.path.expanduser(images_path)
        self.labels_path = os.path.expanduser(labels_path)
        self.output_folder = os.path.expanduser(output_folder)

        # Папки для train, valid, test
        self.train_images_folder = os.path.join(output_folder, "train", "images")
        self.train_labels_folder = os.path.join(output_folder, "train", "labels")
        self.val_images_folder = os.path.join(output_folder, "valid", "images")
        self.val_labels_folder = os.path.join(output_folder, "valid", "labels")
        self.test_images_folder = os.path.join(output_folder, "test", "images")
        self.test_labels_folder = os.path.join(output_folder, "test", "labels")

        # Для хранения изображений и меток
        self.valid_images = []
        self.valid_labels = []
        self.empty_images = []
        self.empty_labels = []

        # Словари для классов по их частоте
        self.label_distribution = Counter()  # Общая статистика меток
        self.train_class_distribution = Counter()
        self.val_class_distribution = Counter()
        self.test_class_distribution = Counter()

    def _prepare_output_folders(self):
        # Создание необходимых папок
        for folder in [
            self.train_images_folder, self.train_labels_folder,
            self.val_images_folder, self.val_labels_folder,
            self.test_images_folder, self.test_labels_folder
        ]:
            os.makedirs(folder, exist_ok=True)

    def analyze_dataset(self):
        # Получение списка файлов с изображениями и метками
        image_files = sorted(f for f in os.listdir(self.images_path) if f.endswith(('.jpg', '.png', '.jpeg')))
        label_files = sorted(f for f in os.listdir(self.labels_path) if f.endswith('.txt'))
        random.shuffle(image_files)

        # Создаем словарь для поиска соответствующих файлов меток, предполагая,
        # что имена файлов до расширений совпадают (без учета папок)
        image_to_label = {os.path.splitext(img)[0]: lbl for img, lbl in zip(sorted(image_files), label_files)}

        # Сортируем файлы меток в соответствии с порядком перемешанного image_files
        label_files = [image_to_label[os.path.splitext(img)[0]] for img in image_files if
                              os.path.splitext(img)[0] in image_to_label]

        # Проверка на соответствие изображений и меток
        assert len(image_files) == len(label_files), "Число изображений и меток не совпадает!"
        assert all(
            Path(image_files[i]).stem == Path(label_files[i]).stem
            for i in range(len(image_files))
        ), "Изображения и метки отличаются по именам!"

        # Фильтрация пустых меток
        for img, lbl in zip(image_files, label_files):
            label_path = os.path.join(self.labels_path, lbl)
            assert Path(img).stem == Path(lbl).stem, "Изображения и метки отличаются по именам!"

            with open(label_path, 'r') as f:
                content = f.read().strip()
            if content:  # Если файл не пустой
                self.valid_images.append(img)
                self.valid_labels.append(lbl)

                # Обновление статистики по классам
                for line in content.splitlines():
                    class_id = line.split()[0]
                    self.label_distribution[class_id] += 1
            else:  # Если файл пустой
                self.empty_images.append(img)
                self.empty_labels.append(lbl)

        # Вывод статистики
        return {
            "total_images": len(image_files),
            "total_labels": len(label_files),
            "valid_labels": len(self.valid_labels),
            "empty_labels": len(self.empty_labels),
            "empty_percentage": f"{len(self.empty_labels) / len(image_files) * 100:.2f}",
            "class_distribution": dict(self.label_distribution)
        }

    def split_dataset(self, train_size=0.75, val_size=0.25, test_size=0, empty_train_percentage=0.0,
                      empty_val_percentage=0.0):

        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "train, val, test ratios must sum up to 1.0"

        # Подготовка папок
        self._prepare_output_folders()

        if test_size == 0:
            # Если тестовый набор не нужен
            train_images, val_images, train_labels, val_labels = train_test_split(
                self.valid_images, self.valid_labels, test_size=(1 - train_size), random_state=42
            )
            test_images, test_labels = [], []  # Пустые списки
        else:
            # Иначе, разделение train и temp (valid + test)
            train_images, temp_images, train_labels, temp_labels = train_test_split(
                self.valid_images, self.valid_labels, test_size=(1 - train_size), random_state=42
            )
            # Разделение temp на valid и test
            val_images, test_images, val_labels, test_labels = train_test_split(
                temp_images, temp_labels, test_size=(test_size / (val_size + test_size)), random_state=42
            )

        # Копирование файлов в папки
        self._copy_files(train_images, self.images_path, self.train_images_folder)
        self._copy_files(train_labels, self.labels_path, self.train_labels_folder)

        self._copy_files(val_images, self.images_path, self.val_images_folder)
        self._copy_files(val_labels, self.labels_path, self.val_labels_folder)

        if test_size > 0:
            self._copy_files(test_images, self.images_path, self.test_images_folder)
            self._copy_files(test_labels, self.labels_path, self.test_labels_folder)

        # Добавление пустых аннотаций, если необходимо (умножаем долю на количество аннотаций)
        num_empty_train_to_add = int(len(train_labels) * empty_train_percentage)
        num_empty_val_to_add = int(len(val_labels) * empty_val_percentage)

        self._add_empty_annotations(num_empty_train_to_add, self.train_images_folder, self.train_labels_folder, "train")
        self._add_empty_annotations(num_empty_val_to_add, self.val_images_folder, self.val_labels_folder, "valid")

        # Сбор статистики по классам для train, val, test
        self.train_class_distribution = self._get_class_distribution(train_labels)
        self.val_class_distribution = self._get_class_distribution(val_labels)

        if test_size > 0:
            self.test_class_distribution = self._get_class_distribution(test_labels)
        else:
            self.test_class_distribution = {}

        # Общая длина всех изображений (для расчета процентных соотношений)
        total_images = len(train_images) + len(val_images) + len(test_images)

        # Расчет процентного соотношения размеров наборов
        train_percentage = (len(train_images) / total_images) * 100 if total_images > 0 else 0
        val_percentage = (len(val_images) / total_images) * 100 if total_images > 0 else 0
        test_percentage = (len(test_images) / total_images) * 100 if total_images > 0 else 0

        # Итоговая статистика
        return {
            "train_class_distribution": dict(self.train_class_distribution),
            "val_class_distribution": dict(self.val_class_distribution),
            "test_class_distribution": dict(self.test_class_distribution),
            "train_size": f"{len(train_images)} ({train_percentage:.2f}%)",
            "val_size": f"{len(val_images)} ({val_percentage:.2f}%)",
            "test_size": f"{len(test_images)} ({test_percentage:.2f}%)",
        }

    def _get_class_distribution(self, labels):
        class_counter = Counter()
        for label_file in labels:
            label_path = os.path.join(self.labels_path, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = line.split()[0]
                    class_counter[class_id] += 1
        return class_counter

    @staticmethod
    def _copy_files(file_list, src_folder, dst_folder):
        for i, file_name in enumerate(file_list):
            shutil.copy(str(os.path.join(src_folder, file_name)), str(os.path.join(dst_folder, file_name)))

    def _add_empty_annotations(self, num_empty_to_add, target_images_folder, target_labels_folder, set_name):
        if num_empty_to_add > len(self.empty_images):
            print(f"Для набора {set_name} не хватает пустых аннотаций. Используем максимум: {len(self.empty_images)}")
            num_empty_to_add = len(self.empty_images)

        if set_name=="valid":
            self.empty_images.reverse()
            self.empty_labels.reverse()

        empty_images_to_add = self.empty_images[:num_empty_to_add]
        empty_labels_to_add = self.empty_labels[:num_empty_to_add]

        self._copy_files(empty_images_to_add, self.images_path, target_images_folder)
        self._copy_files(empty_labels_to_add, self.labels_path, target_labels_folder)

def main():

    parser = argparse.ArgumentParser(
        description="Разделить набор данных train на train, val, и test.")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Путь к исходному набору данных")
    parser.add_argument('--output', type=str, required=True,
                        help="Путь для сохранения результата (train, val, test)")
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help="Пропорция для train набора данных (по умолчанию 70%)")
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="Пропорция для val набора данных (по умолчанию 20%)")
    parser.add_argument('--test_ratio', type=float, default=0,
                        help="Пропорция для test набора данных (по умолчанию 10%)")
    parser.add_argument('--empty_train', type=float, default=1,
                        help="Пропорция для empty_train набора данных (по умолчанию 100%)")

    # Чтение аргументов
    args = parser.parse_args()
    dataset_path = args.dataset
    output_dir = args.output
    test_ratio = args.test_ratio
    val_ratio = args.val_ratio
    train_ratio = args.train_ratio
    empty_train = args.empty_train

    logging.info("Running split dataset script...")
    logging.info(f"Params: ")
    logging.info(f"dataset= {dataset_path}")
    logging.info(f"output= {output_dir}")
    logging.info(f"train_ratio= {train_ratio}")
    logging.info(f"val_ratio= {val_ratio}")
    logging.info(f"test_ratio= {test_ratio}")
    logging.info(f"empty_train= {empty_train}")

    images_folder = os.path.join(dataset_path, "images")
    labels_folder = os.path.join(dataset_path, "labels")


    manager = DatasetManager(images_folder, labels_folder, output_dir)

    # Анализ датасета
    stats = manager.analyze_dataset()
    logging.info("Статистика по входному набору данных:")
    for key, value in stats.items():
        logging.info(f"{key}: {value}")

    # Разделение и добавление пустых меток в train и valid (для valid опционально)
    split_stats = manager.split_dataset(
        train_size=train_ratio, val_size=val_ratio, test_size=test_ratio, empty_train_percentage=empty_train, empty_val_percentage=0
    )
    logging.info("Статистика по разделению:")
    for key, value in split_stats.items():
        logging.info(f"{key}: {value}")

    print_statistics(output_dir, logging)

# Пример использования
if __name__ == "__main__":
    main()
