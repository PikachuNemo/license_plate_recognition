import ast
import cv2
import numpy as np
import pandas as pd
import os

def visualize_video(results_file_path: str, original_video_path: str, output_video_path: str):
    """
    Visualizes the license plate recognition results on the video.

    Args:
        results_file_path (str): Path to the CSV file containing the interpolated results.
        original_video_path (str): Path to the original input video file.
        output_video_path (str): Path to save the processed video with visualizations.
    """
    print(f"[visualize_video] Starting visualization for {original_video_path} with results from {results_file_path}")
    try:
        results = pd.read_csv(results_file_path)
        print(f"[visualize_video] Successfully loaded results from {results_file_path}. Total entries: {len(results)}")
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
        return
    except Exception as e:
        print(f"Error reading results CSV: {e}")
        return

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open original video file at {original_video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'AVC1')  # Changed to AVC1 for broader browser compatibility
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_video_path}. Check codec or file path.")
        cap.release()
        return

    frame_number = 0 # Initialize frame_number to 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video capture to the beginning

    print("[visualize_video] Starting main video frame processing loop...")
    while True:
        frame_number += 1 # Increment frame_number at the beginning of the loop
        ret, frame = cap.read()
        if not ret:
            print(f"[visualize_video] End of video or read error at frame {frame_number-1}. Exiting loop.") # Adjust frame_number for print
            break

        current_frame_results = results[results['frame_number'] == frame_number]

        for _, row in current_frame_results.iterrows():
            try:
                car_bbox_str = row['car_bbox']
                license_plate_bbox_str = row['license_plate_bbox']
                license_plate_number = str(row['license_number']) # Use interpolated license number
                
                # Draw car bounding box
                if car_bbox_str:
                    car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox_str)
                    car_x1, car_y1, car_x2, car_y2 = int(car_x1), int(car_y1), int(car_x2), int(car_y2)
                    cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 5) # Green, thickness 5

                # Draw license plate bounding box
                if license_plate_bbox_str:
                    lp_x1, lp_y1, lp_x2, lp_y2 = ast.literal_eval(license_plate_bbox_str)
                    lp_x1, lp_y1, lp_x2, lp_y2 = int(lp_x1), int(lp_y1), int(lp_x2), int(lp_y2)
                    cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 12) # Red, thickness 12

                    # Place text below the license plate bounding box
                    text_y = lp_y2 + 30 # 30 pixels below license plate bbox
                    text_x = lp_x1
                    
                    # Ensure text fits
                    if text_y < height:
                        cv2.putText(frame, license_plate_number, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except (ValueError, SyntaxError) as e:
                print(f"[visualize_video] Warning: Error processing bbox for frame {frame_number}, car_id {row.get('car_id', 'N/A')}: {e}")
            except Exception as e:
                print(f"[visualize_video] Warning: An unexpected error occurred during visualization for frame {frame_number}, car_id {row.get('car_id', 'N/A')}: {e}")

        out.write(frame)

    cap.release()
    out.release()
    print(f"[visualize_video] Processed video saved to {output_video_path}")
