from ultralytics import YOLO
import cv2 

import utils
from utils import get_car, read_license_plate, write_csv   

from sort.sort import *

import easyocr

mot_tracker = Sort()  # initialize SORT tracker
results = {}  # dictionary to store results for each frame

# loading models
coco_model = YOLO('C:/Users/Nemo/Desktop/license_plate_recognition/plate_detection/yolov8n.pt') 	# for detecting vehicles
license_plate_model = YOLO('C:/Users/Nemo/Desktop/license_plate_recognition/plate_detection/best_t4.pt') 		# for detecting license plates

# load video
cap = cv2.VideoCapture('C:/Users/Nemo/Desktop/license_plate_recognition/videos/plate_test.mp4')

vehicles = [2, 3, 5, 7]  # class ids for car, motorcycle, bus, truck



# read frames
frame_number = -1
ret = True
while ret:
    frame_number += 1
    ret, frame = cap.read()
    if ret and frame_number < 10: 
        results[frame_number] = {}
        
        # detect vehicles 
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf_score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, conf_score])
                
        
        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        # detect license plate
        license_plates = license_plate_model(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, conf_score, class_id = license_plate
            
        
        # assign license plate to vehicles, using car as reference
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        
        # crop license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        
        # process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)  # apply thresholding
        
        # cv2.imshow('original_crop', license_plate_crop)
        # cv2.imshow('threshold', license_plate_crop_thresh)
        # cv2.waitKey(0)
        
        
        # detect license plate lines
        # ocr_results = reader.readtext(license_plate_crop_thresh, detail=1)
        # plate_lines = utils.detect_plate_lines(license_plate_crop_thresh, ocr_results)
        
        
        # read license plate number 
        reader = easyocr.Reader(['ne','en'], gpu=True)
        license_plate_text, license_plate_text_conf_score = read_license_plate(license_plate_crop_thresh, reader)
        
        if license_plate_text is not None:
            results[frame_number][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                          'license_plate': {'bbox': [x1, y1, x2, y2],
                                                            'text': license_plate_text,
                                                                    'bbox_score': conf_score,
                                                                    'text_score': license_plate_text_conf_score}}

# write results
write_csv(results, './recognition_test/test.csv')

        