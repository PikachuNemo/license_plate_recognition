import argparse
import os
import easyocr
from ultralytics import YOLO
import torch
import pandas as pd # Import pandas

from src.utils import write_csv
from src.video_processing import process_video
# from src.database import DatabaseSession # Import DatabaseSession - Commented out
from add_missing_data import interpolate_bounding_boxes
from visualize import visualize_video

def main(video_path: str, output_dir: str, output_video_path: str, coco_model, license_plate_model, reader):
    """
    Main function to run the license plate recognition system.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the directory to save the output CSV file.
        output_video_path (str): Path to save the processed video with visualizations.
        coco_model: The pre-loaded COCO model.
        license_plate_model: The pre-loaded license plate detection model.
        reader: The pre-loaded EasyOCR reader.
    """
    # Initialize the database session with a specific file path - Commented out
    # db_file_path = os.path.join(output_dir, 'license_plates.db')
    # DatabaseSession(db_file=db_file_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None, None

        print(f"[run.py] Starting video processing for {video_path}...")
        # Process the video and get raw plate data
        raw_plate_data_path = process_video(video_path, coco_model, license_plate_model, reader, output_video_path)

        interpolated_df = None # Initialize to None
        interpolated_output_path = None # Initialize to None

        if raw_plate_data_path and os.path.exists(raw_plate_data_path):
            print(f"[run.py] Reading raw plate data from {raw_plate_data_path}...")
            raw_df = pd.read_csv(raw_plate_data_path)
            print(f"[run.py] Raw results loaded. Total entries: {len(raw_df)}")

            if not raw_df.empty:
                print("[run.py] Interpolating bounding boxes...")
                interpolated_df = interpolate_bounding_boxes(raw_df) # Changed to interpolate_bounding_boxes
                print(f"[run.py] Interpolation completed. Total interpolated entries: {len(interpolated_df)}")

        # Save interpolated results
        if interpolated_df: # Only attempt to write if interpolated_df is not None
            interpolated_output_dir = os.path.join('frontend', 'plate_data', 'interpolated')
            os.makedirs(interpolated_output_dir, exist_ok=True)
            base_name = os.path.basename(video_path)
            name, _ = os.path.splitext(base_name)
            interpolated_filename = f"{name}_interpolated.csv"
            interpolated_output_path = os.path.join(interpolated_output_dir, interpolated_filename)
            
            try:
                # Convert to DataFrame before saving
                interpolated_df = pd.DataFrame(interpolated_df)
                interpolated_df.to_csv(interpolated_output_path, index=False)
                print(f"[run.py] Writing interpolated CSV results to {interpolated_output_path}...")
            except Exception as e:
                print(f"Error writing interpolated CSV results to {interpolated_output_path}: {e}")
            
            # Visualize the video with interpolated results
            print(f"[run.py] Starting video visualization to {output_video_path}...")
            final_video_path = visualize_video(interpolated_output_path, video_path, output_video_path)
            print(f"[run.py] Output video saved to {final_video_path}")
            
            return interpolated_output_path, final_video_path # Return path to interpolated CSV and final video
        else:
            print("[run.py] No data to visualize, returning original video path.")
            return None, video_path # Return None for CSV and original video path if no plates detected

    except Exception as e:
        print(f"An unexpected error occurred in main function: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        pass # DatabaseSession.close() # Close the database connection when done - Commented out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('--video', type=str, default='data/test3sec.mp4', help='Path to the input video file')
    parser.add_argument('--output', type=str, default='results/', help='Path to the output directory')
    parser.add_argument('--output_video', type=str, default='results/output_video.mp4', help='Path to save the processed video with visualizations')
    args = parser.parse_args()

    # Load models since we are running this script directly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading models for direct execution...")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    coco_model = YOLO('src/plate_detection/yolov8n.pt').to(device)
    license_plate_model = YOLO('src/plate_detection/detect_train/train4/weights/last.pt').to(device)
    print("Models loaded.")

    main(args.video, args.output, args.output_video, coco_model, license_plate_model, reader)
