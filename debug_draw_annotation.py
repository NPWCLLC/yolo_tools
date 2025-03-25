import os
from ultralytics.data.utils import visualize_image_annotations

# Использование функции
image_path = os.path.expanduser("~/TRAIN_DATA/MERGE_NFS-PUPS-BOXES/train/images/dataset0_4-5-32-0_30161.jpg") # Путь к изображению
label_path = os.path.expanduser("~/TRAIN_DATA/MERGE_NFS-PUPS-BOXES/train/labels/dataset0_4-5-32-0_30161.txt") # Путь к аннотациям YOLO

label_map = {  # Define the label map with all annotated class labels.
    0: "csl",
    1: "ssl",
}

# Visualize
visualize_image_annotations(
    image_path=image_path,
    txt_path=label_path,
    label_map=label_map,
)