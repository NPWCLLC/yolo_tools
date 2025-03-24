import os

if __name__ == '__main__':
    from ultralytics import YOLO

    # Загрузка модели
    model = YOLO("./SSL-PUPS/train_exp/weights/best.pt")

    # Оценка модели
    datasets = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/dataset_boxes_75-20-5_ssl-pups_yolo_all/data.yaml")
    metrics = model.val(data=datasets, imgsz=512, batch=32, device=[0,1],
        project="SSL-PUPS", name="val_boxes_exp111111",)

    # Метрики точности (для детекций коробок)
    print("mAP50-95 (Box Detection): Средняя точность на IoU от 0.5 до 0.95")
    print("mAP50-95: ", metrics.box.map)
    print("mAP50 (Box Detection): Средняя точность на IoU = 0.5")
    print("mAP50: ", metrics.box.map50)
    print("mAP75 (Box Detection): Средняя точность на IoU = 0.75")
    print("mAP75: ", metrics.box.map75)
    print("mAP50-95 per class (Box Detection): ", metrics.box.maps)

    # Метрики полноты и точности (по классам или усреднённые можно взять через mean_results)
    box_results = metrics.box.mean_results()  # Получить усреднённые результаты для детекций
    print("Mean Box Precision (mp): ", box_results[0])
    print("Mean Box Recall (mr): ", box_results[1])

    # Метрики сегментации (для масок)
    if metrics.task != 'detect':
        print("mAP50-95 (Segmentation): Средняя точность на IoU от 0.5 до 0.95")
        print("mAP50-95 (Segmentation): ", metrics.seg.map)
        print("mAP50 (Segmentation): Средняя точность на IoU = 0.5")
        print("mAP50 (Segmentation): ", metrics.seg.map50)
        print("mAP75 (Segmentation): Средняя точность на IoU = 0.75")
        print("mAP75 (Segmentation): ", metrics.seg.map75)
        print("mAP50-95 per class (Segmentation): ", metrics.seg.maps)

        seg_results = metrics.seg.mean_results()
        print("Mean Segmentation Precision (mp): ", seg_results[0])
        print("Mean Segmentation Recall (mr): ", seg_results[1])

    # Метрики скорости
    print("Inference time (ms): Время на обработку одного кадра")
    print("Inference time (ms): ", metrics.speed['inference'])  # Время на обработку
    print("Frames per second (FPS): Количество кадров, обрабатываемых за секунду")
    print("FPS: ", 1000 / metrics.speed['inference'])  # FPS на основе времени вывода