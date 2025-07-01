import os



if __name__ == '__main__':
    from ultralytics import YOLO, checks
    import torch
    import torchvision
    import os
    from utils import setting_logs

    LOG_FILE = 'valid_model_metrics.log'
    logging = setting_logs(LOG_FILE)

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    checks()

    DEVICE = []
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"torchvision version: {torchvision.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        logging.info(f"Number of devices: {num_devices}")

        for i in range(num_devices):
            DEVICE.append(i)
            logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("No CUDA-compatible devices found. Using CPU.")


    # Загрузка модели
    model = YOLO("C:\\Users\\omen_\\OneDrive\\Desktop\\PV_data\\PV-SEG-TRAIN\\train_pv-seg_yolo11x_\\weights\\best.pt")

    # Оценка модели
    datasets = os.path.expanduser("F:\\Largha\\SPLIT_PV-SEG\\data.yaml")
    metrics = model.val(data=datasets, imgsz=640, batch=1, device=DEVICE,
        project="PV-SEG", name="val_seg",)

    logging.info(f"Model: {model.model_name}")
    logging.info(f"Dataset: {datasets}")

    # Метрики точности (для детекций коробок)
    logging.info("mAP50-95 (Box Detection): Средняя точность на IoU от 0.5 до 0.95")
    logging.info(f"mAP50-95: {metrics.box.map}")
    logging.info("mAP50 (Box Detection): Средняя точность на IoU = 0.5")
    logging.info(f"mAP50: {metrics.box.map50}")
    logging.info("mAP75 (Box Detection): Средняя точность на IoU = 0.75")
    logging.info(f"mAP75: {metrics.box.map75}")
    logging.info(f"mAP50-95 per class (Box Detection): {metrics.box.maps}")

    # Метрики полноты и точности (по классам или усреднённые можно взять через mean_results)
    box_results = metrics.box.mean_results()  # Получить усреднённые результаты для детекций
    logging.info(f"Mean Box Precision (mp): {box_results[0]}")
    logging.info(f"Mean Box Recall (mr): {box_results[1]}")

    # Метрики сегментации (для масок)
    if metrics.task != 'detect':
        logging.info("mAP50-95 (Segmentation): Средняя точность на IoU от 0.5 до 0.95")
        logging.info(f"mAP50-95 (Segmentation):  {metrics.seg.map}")
        logging.info("mAP50 (Segmentation): Средняя точность на IoU = 0.5")
        logging.info(f"mAP50 (Segmentation): {metrics.seg.map50}")
        logging.info(f"mAP75 (Segmentation): Средняя точность на IoU = 0.75")
        logging.info(f"mAP75 (Segmentation): {metrics.seg.map75}")
        logging.info(f"mAP50-95 per class (Segmentation): {metrics.seg.maps}")

        seg_results = metrics.seg.mean_results()
        logging.info(f"Mean Segmentation Precision (mp): {seg_results[0]}")
        logging.info(f"Mean Segmentation Recall (mr): {seg_results[1]}")

    # Метрики скорости
    logging.info("Inference time (ms): Время на обработку одного кадра")
    logging.info(f"Inference time (ms): {metrics.speed['inference']}")  # Время на обработку
    logging.info("Frames per second (FPS): Количество кадров, обрабатываемых за секунду")
    logging.info(f"FPS: {1000 / metrics.speed['inference']}")  # FPS на основе времени вывода