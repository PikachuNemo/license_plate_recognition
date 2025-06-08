from ultralytics import YOLO

def main():
    
    # loading a model
    
    # model = YOLO("yolov8n.pt") # using pretrained model as it is faster to train and has better accuracy but it detects objects other than license plates
    
    model = YOLO("config.yaml") # building a new model for license plate detection

    # training the model
    results = model.train(
        data="C:/Users/Nemo/Desktop/license_plate_recognition/plate_detection/config.yaml",
        batch=11, 
        imgsz=640, 
        epochs=11, 
        workers=1,
        project="C:/Users/Nemo/Desktop/license_plate_recognition/plate_detection/detect_train"
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # for Windows compatibility
    main()


