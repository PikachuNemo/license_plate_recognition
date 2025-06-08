from ultralytics import YOLO

# loading a model
model = YOLO("C:/Users/Nemo/Desktop/license_plate_recognition/plate_detection/best_t4.pt") # building a new model

# using the model
model.predict(source="C:/Users/Nemo/Desktop/license_plate_recognition/videos/nepali.mp4", show=True, save=True, conf=0.5, save_txt=True)


