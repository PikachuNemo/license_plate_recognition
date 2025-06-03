from ultralytics import YOLO

# loading a model
model = YOLO("yolov8n.yaml") # building a new model

# using the model
results = model.train(data="C:/Users/Nemo/anaconda_projects/ml/license_plate_recognition/plate_detection/config.yaml", epochs=1)  # train the model




