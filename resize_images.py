import os.path
from pathlib import Path
from PIL import Image


def resize_images_in_folder(input_folder, output_folder, target_size):
    """
    Resize all images in the input_folder to the target_size while maintaining aspect ratio.
    Saves the resized images to the output_folder.

    :param input_folder: Path to the folder containing images.
    :param output_folder: Path to the folder where resized images will be saved.
    :param target_size: Tuple (width, height) of target size.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)  # Создаем директорию, если её нет

    for img_path in input_path.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:  # Фильтруем по форматам
            with Image.open(img_path) as img:
                img.thumbnail(target_size, Image.Resampling.LANCZOS)  # Приведение к размеру с сохранением пропорций
                output_image_path = output_path / img_path.name
                img.save(output_image_path)


# Укажите параметры
input_folder = ""  # Входящая папка с изображениями
output_folder = os.path.join(Path(input_folder).parent,"resized_images")  # Папка для сохранения результата
target_size = (640, 640)  # Целевой размер, например

resize_images_in_folder(input_folder, output_folder, target_size)