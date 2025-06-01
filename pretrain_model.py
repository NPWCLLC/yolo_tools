import random
from pathlib import Path

import lightly_train
import supervision as sv
import yaml
from matplotlib import pyplot as plt
from ultralytics.data.utils import check_det_dataset

if __name__ == "__main__":
    # check_det_dataset
    is_check_dataset = False
    if is_check_dataset:
        dataset = check_det_dataset("C:\\Users\\omen_\\OneDrive\\Desktop\\SSL-CSL\\BALANCE_CLASSES_SSL-CSL-Segm.v7i.yolov11_t80-v20\\data.yaml")

        detections = sv.DetectionDataset.from_yolo(
            data_yaml_path=dataset["yaml_file"],
            images_directory_path=f"{dataset['val']}",
            annotations_directory_path=f"{Path(dataset['val']).parent / 'labels'}",
        )

        with open(dataset["yaml_file"], "r") as f:
            data = yaml.safe_load(f)

        names = data["names"]

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax = ax.flatten()

        detections = [detections[random.randint(0, len(detections))] for _ in range(4)]

        for i, (path, image, annotation) in enumerate(detections):
            annotated_image = box_annotator.annotate(scene=image, detections=annotation)
            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=annotation,
                labels=[names[elem] for elem in annotation.class_id],
            )
            ax[i].imshow(annotated_image[..., ::-1])
            ax[i].axis("off")

        fig.tight_layout()
        fig.show()

    # run pretraining
    data_dir = ""
    lightly_train.train(
        out="pre_train_out/my_experiment",
        data=data_dir,
        model="ultralytics/yolo11x-seg.yaml",
        epochs=100,  # Adjust epochs for faster training.
        batch_size=8,  # Adjust batch size based on hardware.
    )