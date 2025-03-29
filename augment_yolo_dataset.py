import os
import cv2
import glob
from collections import Counter, defaultdict


class YoloDatasetAugmentor:
    TRANSFORMATIONS = [
        (False, 0),  # Оригинальное изображение
        (False, 1),  # Повернуть на 90 градусов
        (False, 2),  # Повернуть на 180 градусов
        (False, 3),  # Повернуть на 270 градусов
        (True, 0),  # Зеркально отразить
        (True, 1),  # Зеркально отразить и повернуть на 90
        (True, 2),  # Зеркально отразить и повернуть на 180
        (True, 3)  # Зеркально отразить и повернуть на 270
    ]

    def __init__(self, dataset_path, output_path, mode="bboxes"):
        """
        Инициализация.

        :param dataset_path: Путь к корневой папке датасета (содержит train, valid, test папки).
        :param output_path: Путь для записи нового аугментированного датасета.
        :param mode: Тип датасета. "bboxes" — обработка боксов, "contours" — обработка контуров.
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.subsets = ["train", "valid", "test"]
        self.mode = mode
        self.class_distributions = defaultdict(Counter)
        self.empty_labels = defaultdict(list)
        self.valid_labels = defaultdict(list)

        if mode not in ["bboxes", "contours"]:
            raise ValueError(f"Unsupported mode: {mode}. Use 'bboxes' or 'contours'.")

    def _prepare_output_directory(self):
        """
        Создание структуры директорий для нового датасета.
        """
        for subset in self.subsets:
            images_dir = os.path.join(self.output_path, subset, "images")
            labels_dir = os.path.join(self.output_path, subset, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

    def dataset_statistics(self, path):
        """
        Вывод статистики по датасету.
        """
        stats = {}
        total_images = 0

        for subset in self.subsets:
            images_dir = os.path.join(path, subset, "images")
            labels_dir = os.path.join(path, subset, "labels")

            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
                subset_class_distribution = self.class_distributions[subset]
                total_images += len(image_files)

                empty_labels_count = len(self.empty_labels[subset])
                valid_labels_count = len(self.valid_labels[subset])

                stats[f"{subset}_size"] = len(image_files)
                stats[f"{subset}_class_distribution"] = dict(subset_class_distribution)
                stats[f"{subset}_valid_labels"] = valid_labels_count
                stats[f"{subset}_empty_labels"] = empty_labels_count
                stats[f"{subset}_empty_percentage"] = f"{(empty_labels_count / len(image_files) * 100):.2f}" \
                    if image_files else "0.00%"

        # Процентное распределение по split'ам
        for subset in self.subsets:
            stats[f"{subset}_percentage"] = f"{(stats.get(f'{subset}_size', 0) / total_images * 100):.2f}%" \
                if total_images > 0 else "0.00%"

        return stats

    def analyze(self, path):
        """
        Анализирует текущий датасет для сбора статистики.
        """
        self.class_distributions = defaultdict(Counter)
        self.empty_labels = defaultdict(list)
        self.valid_labels = defaultdict(list)
        for subset in self.subsets:
            images_dir = os.path.join(path, subset, "images")
            labels_dir = os.path.join(path, subset, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                continue

            image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
            for img_file in image_files:
                img_name = os.path.basename(img_file)
                label_file = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

                if self.mode == "bboxes":
                    annotations = self.read_yolo_labels(label_file)
                else:
                    annotations = self.read_segmentation_labels(label_file)

                if annotations:
                    self.valid_labels[subset].append(img_file)
                    for annotation in annotations:
                        cls = annotation[0]
                        self.class_distributions[subset][cls] += 1
                else:
                    self.empty_labels[subset].append(img_file)

    def augment(self):
        """
        Метод для выполнения аугментаций на всем датасете.
        """
        # Создаем структуру директорий для нового датасета
        self._prepare_output_directory()

        self.analyze(self.dataset_path)
        print("Dataset statistics before augmentation:", self.dataset_statistics(self.dataset_path))

        for subset in self.subsets:
            subset_path = os.path.join(self.dataset_path, subset)
            if os.path.exists(subset_path):
                images_dir = os.path.join(subset_path, "images")
                labels_dir = os.path.join(subset_path, "labels")
                output_images_dir = os.path.join(self.output_path, subset, "images")
                output_labels_dir = os.path.join(self.output_path, subset, "labels")
                self.augment_subset(images_dir, labels_dir, output_images_dir, output_labels_dir)

        self.analyze(self.output_path)
        print("Dataset statistics after augmentation:", self.dataset_statistics(self.output_path))

    def augment_subset(self, images_dir, labels_dir, output_images_dir, output_labels_dir):
        """
        Аугментация с сохранением результатов в output_path.

        :param images_dir: Папка с оригинальными изображениями.
        :param labels_dir: Папка с оригинальными аннотациями.
        :param output_images_dir: Папка для сохранения обработанных изображений.
        :param output_labels_dir: Папка для сохранения обработанных аннотаций.
        """
        image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

            # Чтение изображения и аннотации
            img = cv2.imread(img_path)
            if img is None:
                continue

            if self.mode == "bboxes":
                annotations = self.read_yolo_labels(label_path)
            else:
                annotations = self.read_segmentation_labels(label_path)

            # # Копируем оригинал в output_path
            # cv2.imwrite(os.path.join(output_images_dir, img_name), img)
            # if self.mode == "bboxes":
            #     self.write_yolo_labels(annotations, os.path.join(output_labels_dir, img_name.replace(".jpg", ".txt")),
            #                            img)
            # else:
            #     self.write_segmentation_labels(annotations,
            #                                    os.path.join(output_labels_dir, img_name.replace(".jpg", ".txt")))

            # Аугментация изображений и аннотаций
            for idx, (is_mirrored, n90rotation) in enumerate(self.TRANSFORMATIONS):
                transformed_img, transformed_annotations = self.apply_transformation(
                    img.copy(), annotations, is_mirrored, n90rotation
                )

                # Генерируем уникальные имена для аугментированных данных
                new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{idx}.jpg"
                new_label_name = os.path.splitext(new_img_name)[0] + ".txt"

                # Сохраняем обработанные данные
                cv2.imwrite(os.path.join(output_images_dir, new_img_name), transformed_img)
                if self.mode == "bboxes":
                    self.write_yolo_labels(transformed_annotations, os.path.join(output_labels_dir, new_label_name),
                                           transformed_img)
                else:
                    self.write_segmentation_labels(transformed_annotations,
                                                   os.path.join(output_labels_dir, new_label_name))

    def apply_transformation(self, img, annotations, is_mirrored, n90rotation):
        h, w, _ = img.shape
        if is_mirrored:
            if self.mode == "bboxes":
                annotations = self.flip_annotations_bboxes(annotations)
            else:
                annotations = self.flip_annotations_contours(annotations)

        for _ in range(n90rotation):
            if self.mode == "bboxes":
                annotations = self.rotate90clockwise_annotations_bboxes(annotations)
            else:
                annotations = self.rotate90clockwise_annotations_contours(annotations)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        if is_mirrored:
            img = cv2.flip(img, 1)
        return img, annotations

    def flip_annotations_bboxes(self, bboxes):
        """
        Зеркально отражает боксы относительно вертикальной оси для нормализованных данных.
        """
        flipped = []
        for cls, (x, y, w, h) in bboxes:  # x, y — центр, w, h — размеры
            flipped_x = 1 - x
            flipped.append((cls, (flipped_x, y, w, h)))
        return flipped

    def flip_annotations_contours(self, contours):
        """
        Зеркально отражает контуры относительно вертикальной оси для нормализованных данных.
        """
        flipped = []
        for cls, points in contours:
            flipped_points = [(1 - x, y) for x, y in points]
            flipped.append((cls, flipped_points))
        return flipped

    def rotate90clockwise_annotations_bboxes(self, bboxes):
        """
        Поворачивает боксы на 90 градусов по часовой стрелке для нормализованных данных.
        """
        rotated = []
        for cls, (x, y, w, h) in bboxes:
            rotated_x = y
            rotated_y = 1 - x
            rotated.append((cls, (rotated_x, rotated_y, w, h)))
        return rotated

    def rotate90clockwise_annotations_contours(self, contours):
        """
        Поворачивает контуры на 90 градусов по часовой стрелке для нормализованных данных.
        """
        rotated = []
        for cls, points in contours:
            rotated_points = [(y, 1 - x) for x, y in points]
            rotated.append((cls, rotated_points))
        return rotated

    def write_yolo_labels(self, bboxes, label_path, img):
        """
        Записывает нормализованные аннотации боксов в YOLO-формате.
        """
        with open(label_path, "w") as f:
            for cls, (x, y, w, h) in bboxes:
                f.write(f"{int(cls)} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n")

    def write_segmentation_labels(self, contours, label_path):
        """
        Записывает нормализованные контуры в формате: class x1 y1 x2 y2 ... xn yn.
        """
        with open(label_path, "w") as f:
            for cls, points in contours:
                points_str = " ".join(f"{x:.10f} {y:.10f}" for x, y in points)
                f.write(f"{cls} {points_str}\n")

    def read_yolo_labels(self, label_path):
        """
        Чтение YOLO-аннотаций (боксов).
        """
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    data = line.strip().split(" ")
                    if len(data) == 5:  # YOLO формат: class x_center y_center width height
                        cls = int(data[0])  # Класс объекта
                        bboxes.append((cls, tuple(map(float, data[1:]))))
        return bboxes

    def read_segmentation_labels(self, label_path):
        """
        Чтение аннотаций сегментации с учётом классов.
        """
        contours = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    _line = list(map(float, line.strip().split(" ")))  # Разбиваем строку
                    cls = int(_line[0])  # Первый элемент — это класс
                    points = _line[1:]  # Остальные элементы — координаты
                    if len(points) % 2 == 0:  # Убедимся, что это пары x, y
                        contour = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                        contours.append((cls, contour))  # Добавляем класс и контур
        return contours


# Пример использования
dataset_path = os.path.expanduser("~/TRAIN_DATA/SSL-CSL-SEG/SSL-CSL-Segm.v4i.yolov11")
output_path = os.path.expanduser("~/TRAIN_DATA/SSL-CSL-SEG/SSL-CSL-Segm.Augmented.v4i.yolov11")

# Для боксов:
# augmentor_bboxes = YoloDatasetAugmentor(dataset_path, output_path, mode="bboxes")
# augmentor_bboxes.augment()

# Для сегментации (если требуется):
# augmentor_contours = YoloDatasetAugmentor(dataset_path, output_path, mode="contours")
# augmentor_contours.augment()

augmentor = YoloDatasetAugmentor(dataset_path, output_path, mode="contours")
augmentor.analyze(output_path)
print(augmentor.dataset_statistics(output_path))

