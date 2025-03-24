if __name__ == '__main__':
    from ultralytics import YOLO, checks
    import torch
    import torchvision
    import os

    DEVICE = []
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of devices: {num_devices}")

        for i in range(num_devices):
            DEVICE.append(i)
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        raise "No CUDA-compatible devices found. Using CPU."

    if not DEVICE:
        raise "No CUDA-compatible devices found. Using CPU."

    checks()

    # Load a model
    model = YOLO("yolov8x.pt")

    datasets = os.path.expanduser("~/TRAIN_DATA/NFS-PUPS-BOX/data.yaml")

    results = model.train(
        data=datasets, epochs=600, imgsz=640, batch=48, device=DEVICE,
        project="NFS-PUPS-BOX", name="train_exp_yolov8", patience=10, save_period=10,
    )
    print(results)
