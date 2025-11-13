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

from utils import get_car, read_license_plate, write_csv
from sort.sort import Sort
from database import DatabaseSession, RecognizedPlate # Import database classes

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

        results = {}
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

        # Initialize SORT tracker
        mot_tracker = Sort()

        # read frames
        frame_nmr = -1
        # The 'ret' variable from the initial read is not used here,
        # the loop's 'ret' will be updated by cap.read()
        try:
            with DatabaseSession.session() as db_session: # Start a database session
                while True: # Loop indefinitely until break
                    frame_nmr += 1
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[video_processing] Warning: cap.read() returned False for frame {frame_nmr}. End of video or read error.")
                        break # Exit loop if no more frames
                    
                    original_frame = frame.copy() # Keep a copy of the original frame for drawing and writing
                    
                    # Resize frame for model inference
                    resized_frame = cv2.resize(frame, (640, 640))
                    
                    results[frame_nmr] = {}
                    
                    # detect license plates on the resized frame
                    license_plates = license_plate_model(resized_frame)[0]
                    
                    detections_ = []
                    for license_plate in license_plates.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate
                        if score > 0.7: # Apply confidence threshold
                            detections_.append([x1, y1, x2, y2, score])

                    # update tracker
                    if len(detections_) > 0:
                        print(f"[video_processing] Detections found at frame {frame_nmr}: {len(detections_)}")
                        track_ids = mot_tracker.update(np.asarray(detections_))

                        for track_id in track_ids:
                            x1_resized, y1_resized, x2_resized, y2_resized, car_id = track_id
                            
                            # Scale coordinates back to original frame dimensions
                            x1 = int(x1_resized * (width / 640))
                            y1 = int(y1_resized * (height / 640))
                            x2 = int(x2_resized * (width / 640))
                            y2 = int(y2_resized * (height / 640))

                            # Clip coordinates to ensure they are within frame bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(width, x2)
                            y2 = min(height, y2)

                            # crop license plate from the original frame
                            license_plate_crop = original_frame[int(y1):int(y2), int(x1):int(x2), :]

                            # Skip processing if the crop is empty
                            if license_plate_crop.size == 0:
                                print(f"[video_processing] Warning: Skipping empty license plate crop at frame {frame_nmr}.")
                                continue

                            # process license plate
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            blurred = cv2.GaussianBlur(license_plate_crop_gray, (5, 5), 0) # Add blur for better thresholding
                            _, license_plate_crop_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                            # read license plate text
                            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh, reader)

                            if license_plate_text is not None:
                                results[frame_nmr][car_id] = {'car': {'bbox': [x1, y1, x2, y2]},
                                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                'text': license_plate_text,
                                                                                'bbox_score': score,
                                                                                'text_score': license_plate_text_score}}
                                # Draw visualizations on the original frame
                                cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(original_frame, f"ID: {int(car_id)} - {license_plate_text}", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                # Store recognized plate data in the database
                                recognized_plate_entry = RecognizedPlate(
                                    timestamp=datetime.now().isoformat(),
                                    video_path=video_path,
                                    frame_number=frame_nmr,
                                    car_id=int(car_id),
                                    license_plate_text=license_plate_text,
                                    license_plate_text_score=float(license_plate_text_score),
                                    license_plate_bbox=json.dumps([x1, y1, x2, y2]) # Store bbox as JSON string
                                )
                                db_session.add(recognized_plate_entry)

                    if video_writer:
                        video_writer.write(original_frame)
        except Exception as e:
            print(f"An error occurred during video processing loop: {e}")
            # Optionally re-raise the exception if you want it to propagate
            # raise
        finally:
            print(f"[video_processing] Finished processing {len(results.keys())} frames.")
            cap.release()
            if video_writer:
                print("[video_processing] Releasing video writer.")
                video_writer.release()
        return results
    finally:
        if temp_dir and os.path.exists(temp_dir):
            print(f"[video_processing] Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)