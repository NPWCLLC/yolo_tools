import os
from ultralytics.data.utils import visualize_image_annotations

# Использование функции
image_path = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/SSL-CSL-Segm.v2i.yolov11_balance-class_weight/train/images/img_213_jpg.rf.0c1ad54c277b6f4a9d4403f5f2238e2c.jpg") # Путь к изображению
label_path = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/SSL-CSL-Segm.v2i.yolov11_balance-class_weight/train/labels/img_213_jpg.rf.0c1ad54c277b6f4a9d4403f5f2238e2c.txt")  # Путь к аннотациям YOLO

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