from ultralytics import YOLO

def main():
    # loading a model
    model = YOLO("yolov8n.pt") # using pretrained model as it is faster to train and has better accuracy

    # training the model
    results = model.train(
        data="C:/Users/Nemo/Desktop/license_plate_recognition/plate_detection/config.yaml",
        batch=7, 
        imgsz=640, 
        epochs=1, 
        workers=1,
        project="C:/Users/Nemo/Desktop/license_plate_recognition/plate_detection/detect_train"
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # for Windows compatibility
    main()


