import argparse
import os
import easyocr
from ultralytics import YOLO
import pandas as pd
import numpy as np
import subprocess

from utils import write_csv
from video_processing import process_video

def interpolate_results(results):
    """
    Interpolates the tracking results to fill in missing frames for each car_id.
    """
    interpolated_data = []
    
    # Convert results dictionary to a list of dictionaries for easier DataFrame conversion
    # Each entry in this list will represent a detected license plate in a frame
    processed_results = []
    for frame_nmr, frame_data in results.items():
        for car_id, detection_data in frame_data.items():
            processed_results.append({
                'frame_number': frame_nmr,
                'car_id': car_id,
                'car_bbox': detection_data['car']['bbox'],
                'license_plate_bbox': detection_data['license_plate']['bbox'],
                'license_number': detection_data['license_plate']['text'],
                'license_number_score': detection_data['license_plate']['text_score'],
                'bbox_score': detection_data['license_plate']['bbox_score']
            })
    
    if not processed_results:
        return pd.DataFrame() # Return empty DataFrame if no results

    df = pd.DataFrame(processed_results)
    
    # Ensure car_id is treated as a categorical or object type for grouping
    df['car_id'] = df['car_id'].astype(int)

    # Sort by car_id and frame_number
    df = df.sort_values(by=['car_id', 'frame_number'])

    for car_id in df['car_id'].unique():
        car_df = df[df['car_id'] == car_id].copy()
        
        # Convert bbox strings to actual lists/tuples if they are strings
        # This is important if they were stored as string representations of lists
        car_df['car_bbox'] = car_df['car_bbox'].apply(lambda x: np.array(x) if isinstance(x, list) else np.array(eval(x)))
        car_df['license_plate_bbox'] = car_df['license_plate_bbox'].apply(lambda x: np.array(x) if isinstance(x, list) else np.array(eval(x)))

        # Set frame_number as index for reindexing
        car_df = car_df.set_index('frame_number')
        
        # Create a full range of frames for this car_id
        min_frame = car_df.index.min()
        max_frame = car_df.index.max()
        full_frame_range = pd.Series(range(min_frame, max_frame + 1), name='frame_number')
        
        # Reindex to fill missing frames
        car_df = car_df.reindex(full_frame_range)
        
        # Forward fill non-bbox data (license number, score, car_id)
        car_df['car_id'] = car_df['car_id'].ffill()
        car_df['license_number'] = car_df['license_number'].ffill()
        car_df['license_number_score'] = car_df['license_number_score'].ffill()
        car_df['bbox_score'] = car_df['bbox_score'].ffill()

        # Expand bbox columns into individual coordinate columns, handling NaNs
        # Replace NaN bbox entries with a list of NaNs for consistent expansion
        car_df['car_bbox'] = car_df['car_bbox'].apply(lambda x: x if isinstance(x, list) else [np.nan]*4)
        car_df['license_plate_bbox'] = car_df['license_plate_bbox'].apply(lambda x: x if isinstance(x, list) else [np.nan]*4)

        car_df[['car_x1', 'car_y1', 'car_x2', 'car_y2']] = pd.DataFrame(car_df['car_bbox'].tolist(), index=car_df.index)
        car_df[['lp_x1', 'lp_y1', 'lp_x2', 'lp_y2']] = pd.DataFrame(car_df['license_plate_bbox'].tolist(), index=car_df.index)

        # Interpolate individual coordinates
        car_df['car_x1'] = car_df['car_x1'].interpolate(method='linear', limit_direction='both')
        car_df['car_y1'] = car_df['car_y1'].interpolate(method='linear', limit_direction='both')
        car_df['car_x2'] = car_df['car_x2'].interpolate(method='linear', limit_direction='both')
        car_df['car_y2'] = car_df['car_y2'].interpolate(method='linear', limit_direction='both')
        
        car_df['lp_x1'] = car_df['lp_x1'].interpolate(method='linear', limit_direction='both')
        car_df['lp_y1'] = car_df['lp_y1'].interpolate(method='linear', limit_direction='both')
        car_df['lp_x2'] = car_df['lp_x2'].interpolate(method='linear', limit_direction='both')
        car_df['lp_y2'] = car_df['lp_y2'].interpolate(method='linear', limit_direction='both')

        # Recombine interpolated coordinates into bbox lists
        car_df['car_bbox'] = car_df[['car_x1', 'car_y1', 'car_x2', 'car_y2']].values.tolist()
        car_df['license_plate_bbox'] = car_df[['lp_x1', 'lp_y1', 'lp_x2', 'lp_y2']].values.tolist()

        # Drop temporary columns
        car_df = car_df.drop(columns=['car_x1', 'car_y1', 'car_x2', 'car_y2', 'lp_x1', 'lp_y1', 'lp_x2', 'lp_y2'])

        # Reset index to make frame_number a column again
        car_df = car_df.reset_index()
        
        interpolated_data.append(car_df)
    
    if interpolated_data:
        final_df = pd.concat(interpolated_data).sort_values(by=['frame_number', 'car_id']).reset_index(drop=True)
        return final_df
    else:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition and Tracking")
    parser.add_argument('--video', type=str, default='data/videos/plate_test.mp4',
                        help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results and output video.')
    args = parser.parse_args()

    # Load models
    # coco_model = YOLO('yolov8n.pt') # Not directly used in process_video with current settings
    license_plate_model = YOLO('models/best.pt') # Assuming best.pt is the license plate detection model
    reader = easyocr.Reader(['en']) # English language for OCR

    # Ensure output directories exist
    os.makedirs(os.path.join('frontend', 'outputs'), exist_ok=True)

    # Define paths
    input_video_path = args.video
    output_video_path = os.path.join('frontend', 'outputs', os.path.basename(input_video_path).replace('.', '_') + '.mp4')
    interpolated_csv_path = os.path.join(args.output_dir, 'interpolated_results', 'test_interpolated.csv')

    print(f"Processing video: {input_video_path}")
    print(f"Output video will be saved to: {output_video_path}")
    print(f"Interpolated CSV will be saved to: {interpolated_csv_path}")

    # Process the video
    raw_results = process_video(input_video_path, None, license_plate_model, reader, output_video_path)

    if raw_results:
        # Interpolate results
        interpolated_df = interpolate_results(raw_results)
        
        # Save interpolated results to CSV
        if not interpolated_df.empty:
            interpolated_df.to_csv(interpolated_csv_path, index=False)
            print(f"Interpolated results saved to: {interpolated_csv_path}")
        else:
            print("No interpolated results to save.")
    else:
        print("Video processing returned no raw results.")

    # Run visualize.py to create the final output video with overlaid information
    # Note: visualize.py currently expects 'results/interpolated_results/test_interpolated.csv'
    # and 'data/videos/plate_test.mp4' as input.
    # We need to ensure visualize.py can take dynamic inputs or adjust its hardcoded paths.
    # For now, we'll assume visualize.py is adjusted or the paths match.
    print("Running visualization script...")
    try:
        subprocess.run([
            "python",
            "scripts/visualize.py",
            "--interpolated_csv", interpolated_csv_path,
            "--video_path", input_video_path,
            "--output_video_path", output_video_path
        ], check=True)
        print("Visualization script completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error running visualization script: {e}")
    except FileNotFoundError:
        print("Error: 'python' command not found or scripts/visualize.py not found.")


if __name__ == "__main__":
    main()
