import os
from collections import Counter

# Путь к директории вашего датасета
labels_path = os.path.expanduser("~/TRAIN_DATA/SSL-CSL-SEG/SSL-CSL-Segm.Augmented.v4i.yolov11/train/labels") # Убедитесь, что указана ПАПКА С .txt файлы

# Подсчет экземпляров каждого класса
class_counts = Counter()

for label_file in os.listdir(labels_path):
    if label_file.endswith('.txt'):  # Проверка, чтобы обработать только .txt файлы
        with open(os.path.join(labels_path, label_file), 'r', encoding='utf-8') as file:  # Указание кодировки UTF-8
            for line in file.readlines():
                class_id = int(line.split()[0])  # Первый элемент в строке - ID класса
                class_counts[class_id] += 1

print("Распределение классов:", class_counts)


class_counts = class_counts  # Количество для каждого класса
total_samples = sum(class_counts.values())
num_classes = len(class_counts)

# Рассчитать веса классов
class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
print("Веса классов:", class_weights)

