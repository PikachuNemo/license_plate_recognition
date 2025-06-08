from ultralytics import YOLO
import cv2

# loading models
coco_model = YOLO('yolov8n.pt') 	# for detecting vehicles
license_plate_model = YOLO('.pt') 		# for detecting license plates

# load video
cap = cv2.VideoCapture('./.mp4')

vehicles = [2, 3, 5, 7]  # class ids for car, motorcycle, bus, truck



# read frames
ret = True
while ret:
    frame_number += 1
    ret, frame = cap.read()
    if ret and frame_number < 10: 
        # detect vehicles 
        detections = coco_model(frame)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                
        
        # track vehicles
        
        # detect license plate
        
        # # assign license plate to car
        
        # crop license plate
        
        # process license plate
        
        # read license plate number
        
        
        # write results





















