import os
import cv2
import numpy as np
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import random
import uuid


class YoloDatasetBalancer:
    """
    Класс для балансировки классов в YOLO датасете путем копирования объектов
    из недостаточно представленных классов на пустые изображения.
    """

    def __init__(self, dataset_path, empty_images_path, output_path):
        """
        Инициализация балансировщика датасета.

        :param dataset_path: Путь к корневой папке датасета (содержит train, valid папки)
        :param empty_images_path: Путь к папке с пустыми изображениями для вставки объектов
        :param output_path: Путь для сохранения сбалансированного датасета
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.empty_images_path = os.path.abspath(empty_images_path)
        self.output_path = os.path.abspath(output_path)

        # Статистика датасета
        self.train_stats = {
            'images': 0,
            'annotations': 0,
            'empty_annotations': 0,
            'class_distribution': Counter()
        }

        self.valid_stats = {
            'images': 0,
            'annotations': 0,
            'empty_annotations': 0,
            'class_distribution': Counter()
        }

        # Хранение объектов по классам
        self.train_objects_by_class = defaultdict(list)
        self.valid_objects_by_class = defaultdict(list)

        # Пустые изображения
        self.empty_images = []

    def _prepare_output_directory(self):
        """
        Создание структуры директорий для сбалансированного датасета.
        """
        for subset in ['train', 'valid']:
            images_dir = os.path.join(self.output_path, subset, 'images')
            labels_dir = os.path.join(self.output_path, subset, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

    def analyze_dataset(self):
        """
        Анализ датасета: подсчет изображений, аннотаций, распределения классов.
        """
        # Анализ тренировочного набора
        self._analyze_subset('train')

        # Анализ валидационного набора
        self._analyze_subset('valid')

        # Загрузка пустых изображений
        self._load_empty_images()

        # Вывод статистики
        self._print_dataset_statistics()

    def _analyze_subset(self, subset):
        """
        Анализ подмножества датасета (train или valid).

        :param subset: Название подмножества ('train' или 'valid')
        """
        images_dir = os.path.join(self.dataset_path, subset, 'images')
        labels_dir = os.path.join(self.dataset_path, subset, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Предупреждение: Директория {subset} не найдена")
            return

        # Получаем список файлов
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Статистика для текущего подмножества
        stats = self.train_stats if subset == 'train' else self.valid_stats
        objects_by_class = self.train_objects_by_class if subset == 'train' else self.valid_objects_by_class

        stats['images'] = len(image_files)

        # Анализ аннотаций
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')

            if not os.path.exists(label_file):
                stats['empty_annotations'] += 1
                continue

            # Чтение аннотаций
            with open(label_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                stats['empty_annotations'] += 1
                continue

            # Обработка аннотаций
            for line in lines:
                data = line.strip().split()
                if len(data) < 5:  # Минимум: класс + 2 точки (x, y)
                    continue

                class_id = int(data[0])
                stats['class_distribution'][class_id] += 1
                stats['annotations'] += 1

                # Сохраняем объект для последующего использования
                objects_by_class[class_id].append({
                    'image_path': img_path,
                    'annotation': line.strip(),
                    'points': [float(x) for x in data[1:]]  # Координаты контура
                })

    def _load_empty_images(self):
        """
        Загрузка пустых изображений для вставки объектов.
        """
        if not os.path.exists(self.empty_images_path):
            print(f"Предупреждение: Директория с пустыми изображениями не найдена: {self.empty_images_path}")
            return

        self.empty_images = [
            os.path.join(self.empty_images_path, f) 
            for f in os.listdir(self.empty_images_path) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]

        print(f"Загружено {len(self.empty_images)} пустых изображений")

    def _print_dataset_statistics(self):
        """
        Вывод статистики датасета.
        """
        total_images = self.train_stats['images'] + self.valid_stats['images']
        total_annotations = self.train_stats['annotations'] + self.valid_stats['annotations']

        print("\n=== Статистика датасета ===")
        print(f"Всего изображений: {total_images}")
        print(f"  - Тренировочных: {self.train_stats['images']} ({(self.train_stats['images']/total_images*100 if total_images > 0 else 0):.2f}%)")
        print(f"  - Валидационных: {self.valid_stats['images']} ({(self.valid_stats['images']/total_images*100 if total_images > 0 else 0):.2f}%)")

        print(f"\nВсего аннотаций: {total_annotations}")
        print(f"  - В тренировочном наборе: {self.train_stats['annotations']}")
        print(f"  - В валидационном наборе: {self.valid_stats['annotations']}")

        print(f"\nПустые аннотации:")
        print(f"  - В тренировочном наборе: {self.train_stats['empty_annotations']} ({(self.train_stats['empty_annotations']/self.train_stats['images']*100 if self.train_stats['images'] > 0 else 0):.2f}%)")
        print(f"  - В валидационном наборе: {self.valid_stats['empty_annotations']} ({(self.valid_stats['empty_annotations']/self.valid_stats['images']*100 if self.valid_stats['images'] > 0 else 0):.2f}%)")

        print("\nРаспределение классов в тренировочном наборе:")
        for class_id, count in sorted(self.train_stats['class_distribution'].items()):
            print(f"  - Класс {class_id}: {count} аннотаций")

        print("\nРаспределение классов в валидационном наборе:")
        for class_id, count in sorted(self.valid_stats['class_distribution'].items()):
            print(f"  - Класс {class_id}: {count} аннотаций")

    def balance_classes(self):
        """
        Балансировка классов путем копирования объектов на пустые изображения.
        """
        # Подготовка выходных директорий
        self._prepare_output_directory()

        # Копирование исходного датасета
        self._copy_original_dataset()

        # Балансировка тренировочного набора
        self._balance_subset('train')

        # Балансировка валидационного набора
        self._balance_subset('valid')

        # Анализ сбалансированного датасета
        print("\n=== Анализ сбалансированного датасета ===")
        balanced_balancer = YoloDatasetBalancer(self.output_path, self.empty_images_path, self.output_path)
        balanced_balancer.analyze_dataset()

    def _copy_original_dataset(self):
        """
        Копирование исходного датасета в выходную директорию.
        """
        for subset in ['train', 'valid']:
            src_images_dir = os.path.join(self.dataset_path, subset, 'images')
            src_labels_dir = os.path.join(self.dataset_path, subset, 'labels')

            dst_images_dir = os.path.join(self.output_path, subset, 'images')
            dst_labels_dir = os.path.join(self.output_path, subset, 'labels')

            if not os.path.exists(src_images_dir) or not os.path.exists(src_labels_dir):
                continue

            # Копирование изображений
            for img_file in os.listdir(src_images_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(
                        os.path.join(src_images_dir, img_file),
                        os.path.join(dst_images_dir, img_file)
                    )

            # Копирование аннотаций
            for label_file in os.listdir(src_labels_dir):
                if label_file.endswith('.txt'):
                    shutil.copy2(
                        os.path.join(src_labels_dir, label_file),
                        os.path.join(dst_labels_dir, label_file)
                    )

    def _balance_subset(self, subset):
        """
        Балансировка подмножества датасета.

        :param subset: Название подмножества ('train' или 'valid')
        """
        stats = self.train_stats if subset == 'train' else self.valid_stats
        objects_by_class = self.train_objects_by_class if subset == 'train' else self.valid_objects_by_class

        # Определение максимального количества аннотаций для класса
        max_annotations = max(stats['class_distribution'].values()) if stats['class_distribution'] else 0

        if max_annotations == 0:
            print(f"Предупреждение: В наборе {subset} нет аннотаций для балансировки")
            return

        # Выходные директории
        output_images_dir = os.path.join(self.output_path, subset, 'images')
        output_labels_dir = os.path.join(self.output_path, subset, 'labels')

        # Балансировка каждого класса
        for class_id, count in stats['class_distribution'].items():
            # Сколько аннотаций нужно добавить
            annotations_to_add = max_annotations - count

            if annotations_to_add <= 0:
                continue  # Этот класс уже имеет максимальное количество аннотаций

            print(f"Балансировка класса {class_id} в наборе {subset}: добавление {annotations_to_add} аннотаций")

            # Объекты текущего класса
            class_objects = objects_by_class[class_id]

            if not class_objects:
                print(f"Предупреждение: Нет объектов класса {class_id} для копирования")
                continue

            # Копирование объектов на пустые изображения
            annotations_added = 0
            while annotations_added < annotations_to_add:
                if not self.empty_images:
                    print("Предупреждение: Нет доступных пустых изображений для вставки объектов")
                    break

                # Выбор случайного пустого изображения
                empty_img_path = random.choice(self.empty_images)

                # Определяем случайное количество объектов для добавления (от 1 до 10)
                num_objects = random.randint(1, 10)
                # Ограничиваем количество объектов оставшимся количеством аннотаций для добавления
                num_objects = min(num_objects, annotations_to_add - annotations_added)

                # Выбираем случайные объекты для копирования
                objects_to_add = [random.choice(class_objects) for _ in range(num_objects)]

                # Копирование нескольких объектов на пустое изображение
                self._copy_multiple_objects_to_empty_image(objects_to_add, empty_img_path, output_images_dir, output_labels_dir)

                # Увеличиваем счетчик добавленных аннотаций
                annotations_added += num_objects

    def _copy_multiple_objects_to_empty_image(self, objects, empty_img_path, output_images_dir, output_labels_dir):
        """
        Копирование нескольких объектов на пустое изображение.

        :param objects: Список объектов для копирования (словари с image_path, annotation, points)
        :param empty_img_path: Путь к пустому изображению
        :param output_images_dir: Директория для сохранения результата
        :param output_labels_dir: Директория для сохранения аннотации
        """
        # Чтение пустого изображения
        empty_img = cv2.imread(empty_img_path)

        if empty_img is None:
            print(f"Ошибка при чтении изображения: {empty_img_path}")
            return

        # Создание уникального имени для нового изображения и проверка на существование
        while True:
            # Используем полный UUID для максимальной уникальности
            unique_suffix = f"balanced_{str(uuid.uuid4())}"
            new_img_name = f"{unique_suffix}.jpg"
            new_label_name = f"{unique_suffix}.txt"

            # Пути для сохранения
            new_img_path = os.path.join(output_images_dir, new_img_name)
            new_label_path = os.path.join(output_labels_dir, new_label_name)

            # Проверяем, существуют ли уже файлы с такими именами
            if not os.path.exists(new_img_path) and not os.path.exists(new_label_path):
                break

        # Создаем копию пустого изображения
        result_img = empty_img.copy()
        h_dst, w_dst, _ = result_img.shape

        # Список для хранения всех аннотаций
        all_annotations = []

        # Обрабатываем каждый объект
        for obj in objects:
            # Чтение изображения с объектом
            src_img = cv2.imread(obj['image_path'])
            if src_img is None:
                print(f"Ошибка при чтении изображения: {obj['image_path']}")
                continue

            # Извлечение контура объекта
            data = obj['annotation'].split()
            class_id = int(data[0])
            points = obj['points']

            # Преобразование нормализованных координат в абсолютные
            h_src, w_src, _ = src_img.shape

            # Создаем маску для извлечения объекта
            contour_points = []
            for i in range(0, len(points), 2):
                x = int(points[i] * w_src)
                y = int(points[i + 1] * h_src)
                contour_points.append([x, y])

            contour = np.array(contour_points, dtype=np.int32)
            mask = np.zeros((h_src, w_src), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)

            # Извлечение объекта
            object_roi = cv2.bitwise_and(src_img, src_img, mask=mask)

            # Определение границ объекта
            x, y, w, h = cv2.boundingRect(contour)

            # Вырезаем объект с его границами
            object_cropped = object_roi[y:y+h, x:x+w]
            mask_cropped = mask[y:y+h, x:x+w]

            # Случайное положение для вставки объекта
            max_x = w_dst - w
            max_y = h_dst - h

            if max_x <= 0 or max_y <= 0:
                print(f"Предупреждение: Объект слишком большой для вставки в пустое изображение")
                continue

            # Выбираем случайное положение
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Вставляем объект
            mask_h, mask_w = mask_cropped.shape
            obj_h, obj_w = object_cropped.shape[:2]
            result_h, result_w = result_img.shape[:2]

            for i in range(min(h, mask_h, obj_h)):
                if paste_y + i >= result_h:
                    continue
                for j in range(min(w, mask_w, obj_w)):
                    if paste_x + j >= result_w:
                        continue
                    if mask_cropped[i, j] > 0:  # Если пиксель принадлежит объекту
                        result_img[paste_y + i, paste_x + j] = object_cropped[i, j]

            # Обновляем аннотацию с новыми координатами
            new_contour_points = []
            for point in contour_points:
                # Смещаем координаты относительно нового положения
                new_x = point[0] - x + paste_x
                new_y = point[1] - y + paste_y

                # Нормализуем координаты для нового изображения
                new_x_norm = new_x / w_dst
                new_y_norm = new_y / h_dst

                new_contour_points.append(new_x_norm)
                new_contour_points.append(new_y_norm)

            # Формируем новую строку аннотации
            new_annotation = f"{class_id} " + " ".join([f"{p:.6f}" for p in new_contour_points])
            all_annotations.append(new_annotation)

        # Сохраняем результат
        cv2.imwrite(new_img_path, result_img)

        # Сохраняем все аннотации
        with open(new_label_path, 'w') as f:
            for annotation in all_annotations:
                f.write(annotation + '\n')

        print(f"Создано новое изображение с {len(objects)} объектами: {new_img_path}")

    def _copy_object_to_empty_image(self, obj, empty_img_path, output_images_dir, output_labels_dir):
        """
        Копирование объекта на пустое изображение.

        :param obj: Объект для копирования (словарь с image_path, annotation, points)
        :param empty_img_path: Путь к пустому изображению
        :param output_images_dir: Директория для сохранения результата
        :param output_labels_dir: Директория для сохранения аннотации
        """
        # Используем метод для копирования нескольких объектов, передавая один объект в списке
        self._copy_multiple_objects_to_empty_image([obj], empty_img_path, output_images_dir, output_labels_dir)


def main():
    parser = argparse.ArgumentParser(description="Балансировка классов в YOLO датасете")
    parser.add_argument("--dataset", required=True, help="Путь к корневой папке датасета")
    parser.add_argument("--empty", required=True, help="Путь к папке с пустыми изображениями")
    parser.add_argument("--output", required=True, help="Путь для сохранения сбалансированного датасета")

    args = parser.parse_args()

    balancer = YoloDatasetBalancer(args.dataset, args.empty_images, args.output)
    balancer.analyze_dataset()
    balancer.balance_classes()


if __name__ == "__main__":
    # Пример использования:
    # python balance_classes.py --dataset "path/to/dataset" --empty "path/to/empty/images" --output "path/to/output"

    # Для тестирования можно раскомментировать следующие строки и указать пути к вашим данным:
    # dataset_path = ""
    # empty_images_path = ""
    # output_path = ""

    # balancer = YoloDatasetBalancer(dataset_path, empty_images_path, output_path)
    # balancer.analyze_dataset()
    # balancer.balance_classes()

    main()
