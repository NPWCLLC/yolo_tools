import os
import shutil
import argparse
import uuid
from pathlib import Path


def generate_empty_yolo_dataset(input_folder, output_folder):
    """
    Создает датасет YOLO из папки с изображениями, генерируя пустые txt файлы для аннотаций.
    
    :param input_folder: Путь к папке с изображениями (включая подпапки)
    :param output_folder: Путь к папке для сохранения датасета YOLO
    """
    # Создаем структуру директорий для выходного датасета
    images_dir = os.path.join(output_folder, "images")
    labels_dir = os.path.join(output_folder, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Поддерживаемые форматы изображений
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Счетчики для статистики
    total_images = 0
    
    # Рекурсивный поиск изображений
    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[-1].lower()
            
            # Проверяем, является ли файл изображением
            if file_extension in image_extensions:
                # Получаем имя файла без расширения
                base_name = Path(file).stem

                unique_suffix = f"empty_data_{str(uuid.uuid4())[:8]}"

                # Пути для сохранения
                dest_image_path = os.path.join(images_dir, f"{unique_suffix}_{file}")
                dest_label_path = os.path.join(labels_dir, f"{Path(dest_image_path).stem}.txt")
                
                # Копируем изображение
                shutil.copy2(file_path, dest_image_path)
                
                # Создаем пустой txt файл
                with open(dest_label_path, 'w') as f:
                    pass  # Создаем пустой файл
                
                total_images += 1
    
    print(f"Обработка завершена!")
    print(f"Всего обработано изображений: {total_images}")
    print(f"Изображения сохранены в: {images_dir}")
    print(f"Пустые аннотации сохранены в: {labels_dir}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Генерация пустых txt файлов для YOLO датасета из папки с изображениями")
    # parser.add_argument("input_folder", help="Путь к папке с изображениями")
    # parser.add_argument("output_folder", help="Путь к папке для сохранения датасета YOLO")
    #
    # args = parser.parse_args()
    
    # Преобразуем пути в абсолютные
    input_folder = os.path.abspath("")
    output_folder = os.path.abspath("")
    
    # Проверяем существование входной папки
    if not os.path.exists(input_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    generate_empty_yolo_dataset(input_folder, output_folder)