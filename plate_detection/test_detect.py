from ultralytics import YOLO

# loading a model
model = YOLO("yolov8n.pt") # building a new model

# using the model
model.predict(source="C:/Users/Nemo/Desktop/license_plate_recognition//videos/video2.mp4", show=True, save=True, conf=0.5, save_txt=True)


