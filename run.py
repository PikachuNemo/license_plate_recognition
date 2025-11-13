import argparse
import os
import easyocr
from ultralytics import YOLO
import torch

from src.utils import write_csv
from src.video_processing import process_video
from src.database import DatabaseSession # Import DatabaseSession

def main(video_path: str, output_dir: str, output_video_path: str):
    """
    Main function to run the license plate recognition system.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the directory to save the output CSV file.
        output_video_path (str): Path to save the processed video with visualizations.
    """
    # Initialize the database session with a specific file path
    db_file_path = os.path.join(output_dir, 'license_plates.db')
    DatabaseSession(db_file=db_file_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    reader = easyocr.Reader(['en', 'ne'], gpu=torch.cuda.is_available())
    coco_model = YOLO('src/plate_detection/yolov8n.pt').to(device)
    license_plate_model = YOLO('models/best.pt').to(device)

    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None

        results = process_video(video_path, coco_model, license_plate_model, reader, output_video_path)

        if results is None: # Check if video processing failed
            print(f"Error: Video processing failed for {video_path}.")
            return None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, 'results.csv')
        try:
            print(f"[main] Results before writing CSV: {results}") # Added print statement
            write_csv(results, output_path)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error writing CSV results to {output_path}: {e}")
            return None        
        print(f"Output video saved to {output_video_path}")
        return output_video_path
    except Exception as e:
        print(f"An unexpected error occurred in main function: {e}")
        return None
    finally:
        DatabaseSession.close() # Close the database connection when done

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('--video', type=str, default='data/videos/plate_test.mp4', help='Path to the input video file')
    parser.add_argument('--output', type=str, default='results/', help='Path to the output directory')
    parser.add_argument('--output_video', type=str, default='results/output_video.mp4', help='Path to save the processed video with visualizations')
    args = parser.parse_args()

    main(args.video, args.output, args.output_video)
