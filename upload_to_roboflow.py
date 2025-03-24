
if __name__ == "__main__":
    import multiprocessing
    import os
    import roboflow

    num_workers = multiprocessing.cpu_count()

    rf = roboflow.Roboflow(api_key="RC1j3CFTGzS6iJETTFlq")

    # get a workspace
    workspace = rf.workspace("sslanumals")
    OUTPUT_DIR = os.path.expanduser("~/DENIS_GAEV/TRAIN/TEMP-GAEV/yolo_dataset_ex2")
    # Upload data set to a new/existing project
    workspace.upload_dataset(
        OUTPUT_DIR, # This is your dataset path
        "ssl-pups", # This will either create or get a dataset with the given ID
        num_workers=num_workers
    )