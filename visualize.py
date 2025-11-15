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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v as requested by the user
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

    license_plate_data = {}
    # Pre-process license plate crops and numbers
    print("[visualize_video] Pre-processing license plate crops and numbers...")
    for car_id in np.unique(results['car_id']):
        car_results = results[results['car_id'] == car_id]
        if car_results.empty:
            continue

        # Find the entry with the highest license_number_score for this car_id
        if car_results['license_number_score'].notna().any():
            max_score_entry = car_results.loc[car_results['license_number_score'].idxmax()]
        else:
            max_score_entry = car_results.iloc[0]

        frame_num_for_crop = max_score_entry['frame_number']
        license_plate_number = str(max_score_entry['license_number'])
        license_plate_bbox_str = max_score_entry['license_plate_bbox']

        # Attempt to read the frame for cropping
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_for_crop)
        ret, frame_for_crop = cap.read()

        license_crop = None
        if ret and license_plate_bbox_str:
            try:
                x1, y1, x2, y2 = ast.literal_eval(license_plate_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Clip coordinates to frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                if x2 > x1 and y2 > y1:
                    license_crop = frame_for_crop[y1:y2, x1:x2]
                    if license_crop.size > 0:
                        # Resize for consistent display
                        license_crop = cv2.resize(license_crop, (150, 75)) # Fixed size for display
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse bbox '{license_plate_bbox_str}' for car_id {car_id} at frame {frame_num_for_crop}: {e}")
            except Exception as e:
                print(f"Warning: Failed to crop license plate for car_id {car_id} at frame {frame_num_for_crop}: {e}")
        else:
            print(f"Warning: Failed to read frame {frame_num_for_crop} or missing bbox for car_id {car_id} during license_crop initialization.")

        license_plate_data[car_id] = {
            'license_crop': license_crop,
            'license_plate_number': license_plate_number
        }
    print("[visualize_video] Finished pre-processing license plate crops.")

    frame_number = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video capture to the beginning

    print("[visualize_video] Starting main video frame processing loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[visualize_video] End of video or read error at frame {frame_number}. Exiting loop.")
            break
        frame_number += 1

        current_frame_results = results[results['frame_number'] == frame_number]

        for _, row in current_frame_results.iterrows():
            try:
                car_bbox_str = row['car_bbox']
                license_plate_bbox_str = row['license_plate_bbox']
                car_id = row['car_id']

                # Draw car bounding box
                if car_bbox_str:
                    car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    car_x1, car_y1, car_x2, car_y2 = int(car_x1), int(car_y1), int(car_x2), int(car_y2)
                    cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 5) # Green, thickness 5

                # Draw license plate bounding box
                if license_plate_bbox_str:
                    lp_x1, lp_y1, lp_x2, lp_y2 = ast.literal_eval(license_plate_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    lp_x1, lp_y1, lp_x2, lp_y2 = int(lp_x1), int(lp_y1), int(lp_x2), int(lp_y2)
                    cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 12) # Red, thickness 12

                # Display cropped license plate and text below the car
                if car_id in license_plate_data:
                    lp_info = license_plate_data[car_id]
                    license_crop = lp_info['license_crop']
                    license_plate_number = lp_info['license_plate_number']

                    if license_crop is not None:
                        crop_h, crop_w, _ = license_crop.shape

                        # Position below the car bounding box
                        display_y = car_y2 + 10 # 10 pixels below car bbox
                        display_x = car_x1

                        # Ensure it fits within the frame
                        if display_y + crop_h < height and display_x + crop_w < width:
                            frame[display_y : display_y + crop_h, display_x : display_x + crop_w] = license_crop
                            cv2.rectangle(frame, (display_x, display_y), (display_x + crop_w, display_y + crop_h), (0, 255, 0), 3) # Green border

                            # Place text below the cropped image
                            text_y = display_y + crop_h + 30 # 30 pixels below cropped image
                            text_x = display_x
                            
                            # Ensure text fits
                            if text_y < height:
                                cv2.putText(frame, license_plate_number, (text_x, text_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        else:
                            print(f"[visualize_video] Warning: Cropped license plate for car_id {car_id} at frame {frame_number} would be out of bounds. Skipping display.")

            except (ValueError, SyntaxError) as e:
                print(f"[visualize_video] Warning: Error processing bbox for frame {frame_number}, car_id {car_id}: {e}")
            except Exception as e:
                print(f"[visualize_video] Warning: An unexpected error occurred during visualization for frame {frame_number}, car_id {car_id}: {e}")

        out.write(frame)

    cap.release()
    out.release()
    print(f"[visualize_video] Processed video saved to {output_video_path}")
