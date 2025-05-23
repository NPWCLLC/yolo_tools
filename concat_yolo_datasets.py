import os
import shutil


def merge_yolo_datasets_to_train(root_dir, output_dir, target_class=None):
    """
    Универсальное решение для объединения нескольких наборов данных YOLO в одну папку train.
    Учитываются структуры с train, val, test и наборы только с train.
    Добавлена возможность объединения всех классов в один.

    :param root_dir: Корневая папка, содержащая поддиректории с датасетами
    :param output_dir: Папка для выходного объединенного набора данных (все объединено в train)
    :param target_class: Объединяемый целевой класс (по умолчанию 0)
    """
    # Создаем выходные папки
    images_output_dir = os.path.join(output_dir, "train", "images")
    labels_output_dir = os.path.join(output_dir, "train", "labels")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    dataset_count = 0  # Уникальный счетчик для каждого набора данных

    # Проходим через все папки в корневом каталоге
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)

        # Проверяем, что это папка
        if not os.path.isdir(dataset_path):
            continue

        # Определяем подкаталоги (train, val, test или один train)
        subsets = ["train", "valid", "test"]  # Все возможные поддиректории
        for subset in subsets:
            subset_path = os.path.join(dataset_path, subset)

            # Если подкаталог существует, проверяем наличие images и labels
            if os.path.exists(subset_path):
                images_path = os.path.join(subset_path, "images")
                labels_path = os.path.join(subset_path, "labels")

                if not os.path.isdir(images_path) or not os.path.isdir(labels_path):
                    print(f"Пропуск: {subset_path} (нет 'images' или 'labels')")
                    continue

                # Уникальный идентификатор для борьбы с конфликтами имен
                dataset_id = f"dataset{dataset_count}"

                # Копируем данные
                for filename in os.listdir(images_path):
                    if filename.endswith((".jpg", ".jpeg", ".png")):
                        base_name, ext = os.path.splitext(filename)
                        new_image_name = f"{dataset_id}_{base_name}{ext}"
                        new_label_name = f"{dataset_id}_{base_name}.txt"

                        # Копируем изображение
                        old_img_path = os.path.join(images_path, filename)
                        new_img_path = os.path.join(images_output_dir, new_image_name)
                        shutil.copy2(old_img_path, new_img_path)

                        # Копируем аннотацию
                        old_label_path = os.path.join(labels_path, f"{base_name}.txt")
                        new_label_path = os.path.join(labels_output_dir, new_label_name)
                        if os.path.exists(old_label_path):
                            # Обрабатываем аннотационный файл
                            with open(old_label_path, "r") as label_file:
                                lines = label_file.readlines()

                            if target_class is not None:
                                # Переписываем содержимое файла, меняя класс на target_class
                                with open(new_label_path, "w") as output_label_file:
                                    for line in lines:
                                        parts = line.strip().split()
                                        if len(parts) > 0:
                                            parts[0] = str(target_class)  # Заменяем номер класса на целевой класс
                                            output_label_file.write(" ".join(parts) + "\n")
                        else:
                            print(f"Пропуск аннотации для {filename} (нет .txt файла)")

        # Увеличиваем счетчик после обработки текущего набора данных
        dataset_count += 1

    print(f"Объединение завершено! Все данные сохранены в {images_output_dir} и {labels_output_dir}")


if __name__ == "__main__":
    # Пример использования
    root_dir = os.path.expanduser(
        "F:\\Largha\\PV_dataset")  # Укажите путь к корневой папке с датасетами
    output_dir = os.path.expanduser(
        "C:\\Users\\omen_\\OneDrive\\Desktop\\MERGE_PV-SEG")  # Укажите путь для сохранения данных (все объединится в train)

    # target_class Задайте целевой класс (например, объединить все классы в класс 0)
    merge_yolo_datasets_to_train(root_dir, output_dir, target_class=0)