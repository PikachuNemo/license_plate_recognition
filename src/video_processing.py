import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import tempfile
import shutil
import subprocess
from datetime import datetime
import json
import pandas as pd # Added pandas import

from .utils import get_car, read_license_plate, write_csv
from .sort.sort import Sort
from .database import DatabaseSession, RecognizedPlate # Import database classes

def process_video(video_path: str, coco_model, license_plate_model, reader, output_video_path: str = None):
    """
    Processes a video to detect and recognize license plates using batch processing for performance.

    Args:
        video_path (str): Path to the input video file.
        coco_model: YOLO model for vehicle detection (not used with current settings).
        license_plate_model: YOLO model for license plate detection.
        reader: EasyOCR reader instance.
        output_video_path (str): Path to save the processed video with visualizations.

    Returns:
        dict: A dictionary containing the detection and recognition results.
    """
    print(f"[video_processing] Starting video processing for: {video_path}")

    temp_dir = None
    processed_video_path = video_path # Default to original path
    raw_plate_data_list = [] # New list to store raw plate data

    try:
        # Initialize VideoCapture with the original video path
        cap = cv2.VideoCapture(video_path)
        print(f"[video_processing] VideoCapture.isOpened() for original video: {cap.isOpened()}")

        # Attempt to read the first frame. If it fails, try re-encoding.
        ret, frame = cap.read()
        if not ret:
            print(f"[video_processing] Initial read of {video_path} failed. Attempting re-encoding with FFmpeg.")
            cap.release() # Release the problematic capture

            temp_dir = tempfile.mkdtemp()
            temp_output_video_path = os.path.join(temp_dir, "reencoded_video.mp4")
            
            ffmpeg_command = [
                "ffmpeg",
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "copy",
                temp_output_video_path
            ]
            
            print(f"[video_processing] Running FFmpeg command: {' '.join(ffmpeg_command)}")
            try:
                result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                print(f"[video_processing] FFmpeg stdout: {result.stdout}")
                print(f"[video_processing] FFmpeg stderr: {result.stderr}")
                processed_video_path = temp_output_video_path
                print(f"[video_processing] Video successfully re-encoded to: {processed_video_path}")
            except subprocess.CalledProcessError as e:
                print(f"[video_processing] FFmpeg re-encoding failed (exit code {e.returncode}): {e.stderr}")
                print(f"[video_processing] Falling back to original video path, but it might not be readable.")
                processed_video_path = video_path # Fallback if re-encoding fails
            except FileNotFoundError:
                print("[video_processing] FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")
                print(f"[video_processing] Falling back to original video path, but it might not be readable.")
                processed_video_path = video_path # Fallback if ffmpeg is not found
        
        print(f"[video_processing] Final video path for OpenCV: {processed_video_path}")
        print(f"[video_processing] Does final video path exist? {os.path.exists(processed_video_path)}")
        cap = cv2.VideoCapture(processed_video_path)
        print(f"[video_processing] VideoCapture.isOpened() for processed video: {cap.isOpened()}")
        
        # Read the first frame again from the potentially new capture
        ret, frame = cap.read()
        if not ret:
            print(f"[video_processing] Error: Could not read first frame from {processed_video_path} even after re-encoding attempt. Video might be corrupted or unreadable.")
            cap.release()
            return None
        
        # If we reached here, 'cap' is open and 'frame' holds the first frame
        # Reset video capture to the beginning for actual processing
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # results = {} # No longer needed as a dict of dicts
        frame_number = 0

        # Initialize video writer
        video_writer = None
        if output_video_path:
            # Use 'mp4v' codec.
            output_video_path = os.path.splitext(output_video_path)[0] + '.mp4' # Ensure .mp4 extension
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"[video_processing] VideoWriter parameters: Path={output_video_path}, FourCC={fourcc}, FPS={fps}, Dimensions=({width}, {height})")
            
            if fps == 0 or width == 0 or height == 0:
                print(f"[video_processing] Warning: Invalid video properties detected. FPS={fps}, Width={width}, Height={height}. Cannot create video writer.")
                return None # Return None if video properties are invalid
            else:
                try:
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if not video_writer.isOpened():
                        raise Exception(f"Failed to open VideoWriter for {output_video_path}. This might be due to missing codecs (e.g., ensure 'ffmpeg' is installed and configured for OpenCV), incorrect path, or permissions issues.")
                except Exception as e:
                    print(f"[video_processing] Exception while creating VideoWriter: {e}")
                    raise
        else: # If output_video_path is not provided, we can't write a video
            return None # Indicate failure to produce an output video

        # Initialize SORT trackers for both cars and license plates
        mot_tracker_cars = Sort()
        mot_tracker_license_plates = Sort()

        # Define vehicle classes for coco_model (e.g., car, truck, bus, motorcycle)
        vehicle_classes = [2, 3, 5, 7] # COCO classes for car, motorcycle, bus, truck

        # read frames
        frame_nmr = -1
        try: # Re-introducing the try block
            # with DatabaseSession.session() as db_session: # Start a database session - Commented out for now
            while True: # Loop indefinitely until break
                frame_nmr += 1
                ret, frame = cap.read()
                if not ret:
                    print(f"[video_processing] Warning: cap.read() returned False for frame {frame_nmr}. End of video or read error.")
                    break # Exit loop if no more frames
                print(f"[video_processing] Processing frame {frame_nmr}")
                
                original_frame = frame.copy() # Keep a copy of the original frame for drawing and writing
                
                # Resize frame for model inference
                resized_frame = cv2.resize(frame, (640, 640))
                
                # results[frame_nmr] = {} # No longer needed
                
                # 1. Detect cars using coco_model
                car_detections = coco_model(resized_frame)[0]
                print(f"[video_processing] Frame {frame_nmr}: Found {len(car_detections.boxes.data.tolist())} raw car detections.")
                cars_to_track = []
                for car_det in car_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = car_det
                    if int(class_id) in vehicle_classes and score > 0.5: # Confidence threshold for cars
                        cars_to_track.append([x1, y1, x2, y2, score])
                
                # Update car tracker
                if len(cars_to_track) > 0:
                    car_track_ids = mot_tracker_cars.update(np.asarray(cars_to_track))
                else:
                    car_track_ids = mot_tracker_cars.update(np.empty((0, 5)))
                print(f"[video_processing] Frame {frame_nmr}: Tracking {len(car_track_ids)} cars.")
                
                # 2. Detect license plates using license_plate_model
                license_plate_detections = license_plate_model(resized_frame)[0]
                print(f"[video_processing] Frame {frame_nmr}: Found {len(license_plate_detections.boxes.data.tolist())} raw license plate detections.")
                license_plates_to_track = []
                for lp_det in license_plate_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = lp_det
                    if score > 0.7: # Confidence threshold for license plates
                        license_plates_to_track.append([x1, y1, x2, y2, score])
                
                # Update license plate tracker
                if len(license_plates_to_track) > 0:
                    license_plate_track_ids = mot_tracker_license_plates.update(np.asarray(license_plates_to_track))
                else:
                    license_plate_track_ids = mot_tracker_license_plates.update(np.empty((0, 5)))
                print(f"[video_processing] Frame {frame_nmr}: Tracking {len(license_plate_track_ids)} license plates.")

                # Store scaled car and license plate bboxes with their IDs
                scaled_cars = []
                for car_track in car_track_ids:
                    x1_c, y1_c, x2_c, y2_c, car_id = car_track
                    scaled_cars.append([
                        int(x1_c * (width / 640)), int(y1_c * (height / 640)),
                        int(x2_c * (width / 640)), int(y2_c * (height / 640)),
                        car_id
                    ])

                for lp_track in license_plate_track_ids:
                    x1_lp, y1_lp, x2_lp, y2_lp, lp_id = lp_track
                    
                    # Scale license plate coordinates
                    x1_lp_scaled = int(x1_lp * (width / 640))
                    y1_lp_scaled = int(y1_lp * (height / 640))
                    x2_lp_scaled = int(x2_lp * (width / 640))
                    y2_lp_scaled = int(y2_lp * (height / 640))

                    # Clip coordinates to ensure they are within frame bounds
                    x1_lp_scaled = max(0, x1_lp_scaled)
                    y1_lp_scaled = max(0, y1_lp_scaled)
                    x2_lp_scaled = min(width, x2_lp_scaled)
                    y2_lp_scaled = min(height, y2_lp_scaled)

                    # Associate license plate with a car
                    print(f"[video_processing] Frame {frame_nmr}, LP ID {lp_id}: Attempting to associate with car.")
                    car_bbox_for_lp, car_id_for_lp = get_car(
                        [x1_lp_scaled, y1_lp_scaled, x2_lp_scaled, y2_lp_scaled],
                        scaled_cars
                    )
                    print(f"[video_processing] Frame {frame_nmr}, LP ID {lp_id}: Associated with Car ID {car_id_for_lp}.")

                    if car_id_for_lp != -1: # If a car is found for this license plate
                        # crop license plate from the original frame
                        license_plate_crop = original_frame[y1_lp_scaled:y2_lp_scaled, x1_lp_scaled:x2_lp_scaled, :]

                        # Skip processing if the crop is empty or malformed
                        if license_plate_crop.size == 0 or license_plate_crop.shape[0] == 0 or license_plate_crop.shape[1] == 0 or license_plate_crop.ndim < 3:
                            print(f"[video_processing] Warning: Skipping malformed license plate crop at frame {frame_nmr} for LP ID {lp_id}. Shape: {license_plate_crop.shape}")
                            continue
                        print(f"[video_processing] Frame {frame_nmr}, LP ID {lp_id}: license_plate_crop shape: {license_plate_crop.shape}")
                        # process license plate
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(license_plate_crop_gray, (5, 5), 0)
                        _, license_plate_crop_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                        # read license plate text
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh, reader)
                        print(f"[video_processing] Frame {frame_nmr}, Car ID {car_id_for_lp}, LP ID {lp_id}: Recognized text '{license_plate_text}' with score {license_plate_text_score}.")

                        if license_plate_text is not None:
                            # Collect raw data for later saving
                            raw_plate_data_list.append({
                                'frame_number': frame_nmr,
                                'car_id': car_id_for_lp,
                                'car_bbox': car_bbox_for_lp,
                                'license_plate_bbox': [x1_lp_scaled, y1_lp_scaled, x2_lp_scaled, y2_lp_scaled],
                                'license_plate_bbox_score': lp_track[4],
                                'license_number': license_plate_text,
                                'license_number_score': license_plate_text_score
                            })

                if video_writer:
                    video_writer.write(original_frame)
        except Exception as e:
            print(f"An error occurred during video processing loop: {e}")
            raise # Re-raise the exception to propagate it
        finally:
            print(f"[video_processing] Finished processing {frame_nmr} frames.")
            cap.release()
            if video_writer:
                print("[video_processing] Releasing video writer.")
                video_writer.release()
            
            # Save raw_plate_data_list to CSV
            if raw_plate_data_list:
                raw_data_output_dir = os.path.join('frontend', 'plate_data', 'raw_plate_data')
                os.makedirs(raw_data_output_dir, exist_ok=True)
                base_name = os.path.basename(video_path)
                name, _ = os.path.splitext(base_name)
                raw_data_filename = f"{name}_raw_plate_data.csv"
                raw_data_output_path = os.path.join(raw_data_output_dir, raw_data_filename)
                
                raw_df = pd.DataFrame(raw_plate_data_list)
                raw_df.to_csv(raw_data_output_path, index=False)
                print(f"[video_processing] Raw plate data saved to {raw_data_output_path}")
                return raw_data_output_path # Return the path to the raw data CSV
            else:
                print("[video_processing] No raw plate data collected.")
                return None
    finally:
        if temp_dir and os.path.exists(temp_dir):
            print(f"[video_processing] Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)