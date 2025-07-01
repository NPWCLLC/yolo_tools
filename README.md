# YOLO Tools for NPWC

This repository contains a collection of utility scripts for working with YOLO (You Only Look Once) object detection and segmentation models. These tools facilitate dataset preparation, augmentation, conversion, training, and validation for computer vision tasks.

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

for pretrained model by lightly-train GPU
```bash
 pip install lightly-train torch torchvision pytorch-lightning --index-url https://download.pytorch.org/whl/cu118
```

## Tools Overview

### Dataset Preparation

#### `augment_yolo_dataset.py`
A tool for augmenting YOLO datasets by applying transformations such as flipping and rotation. Supports both bounding box and segmentation annotations.

#### `balance_classes.py`
Balances class distributions in YOLO datasets by analyzing class frequencies and copying objects to empty images to achieve more balanced training data.

#### `concat_yolo_datasets.py`
Merges multiple YOLO datasets into a single training dataset. Handles different dataset structures and can optionally convert all class labels to a single target class.

#### `convert_unet_dataset_to_yolo.py`
Converts datasets from U-Net format (image and mask pairs) to YOLO format. Can generate either bounding box annotations or segmentation contours.

#### `convert_yolo_seg_to_yolo_box.py`
Converts YOLO segmentation annotations to YOLO bounding box annotations by calculating the minimum bounding box for each segmentation polygon.

#### `generate_empty_data_yolo.py`
Creates a YOLO dataset from a folder of images by generating empty annotation files. Useful for creating background/negative samples or preparing a dataset structure for manual annotation.

#### `resize_images.py`
Resizes all images in a folder to a target size while maintaining aspect ratio. Useful for preparing images for YOLO training.

#### `split_dataset.py`
Splits a dataset into train/validation/test sets and adds a specified percentage of empty (background) images to each set.

### Debugging and Visualization

#### `debug_annotations.py`
Visualizes YOLO segmentation annotations by drawing contours on images. Useful for verifying the correctness of segmentation annotations.

#### `debug_draw_annotation.py`
Provides two approaches for visualizing YOLO annotations: using the Ultralytics library's built-in visualization for bounding boxes and a manual approach for segmentation contours.

### Training and Validation

#### `pretrain_model.py`
Provides functionality for checking/visualizing a detection dataset and running pre-training on a YOLO model using a custom lightly_train module.

#### `train_utils.py`
Contains a custom YOLOWeightedDataset class that extends the Ultralytics YOLODataset to implement weighted sampling for handling class imbalance during training.

#### `train_yolo.py`
A script for training a YOLO model using the Ultralytics library. Checks for CUDA availability, loads a pre-trained model, and trains it on a specified dataset.

#### `valid_model_metrics.py`
Evaluates a trained YOLO model on a validation dataset and prints various metrics including mAP (mean Average Precision), precision, recall, and inference speed.

## Usage Examples

Each script contains example usage at the bottom of the file. To use a script, modify the input/output paths and parameters as needed, then run:

```bash
python script_name.py
```

For most scripts, you'll need to specify:
- Input directory containing images and/or annotations
- Output directory for processed files
- Any specific parameters for the task (e.g., target size for resizing, class IDs for balancing)

## Requirements

The tools in this repository depend on various Python libraries including:
- ultralytics (YOLO implementation)
- opencv-python (image processing)
- numpy (numerical operations)
- matplotlib (visualization)
- PIL/Pillow (image manipulation)
- scikit-learn (for dataset splitting)

See `requirements.txt` for the complete list of dependencies.



