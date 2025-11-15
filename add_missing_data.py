import csv
import numpy as np
from scipy.interpolate import interp1d
import os
from collections import defaultdict
import cv2
import pandas as pd
import ast # Import ast for literal_eval
import re # Import re module

def parse_bbox(bbox_str):
    """
    Parses a string representation of a bounding box into a list of floats.
    Handles various string formats by replacing common delimiters with commas.
    """
    if pd.isna(bbox_str) or bbox_str == '':
        return [-1, -1, -1, -1] # Return invalid bbox for NaN or empty string
    try:
        # Remove brackets and split by comma
        cleaned_str = bbox_str.strip('[]')
        # Handle cases where numbers might be separated by spaces instead of commas
        if ' ' in cleaned_str and ',' not in cleaned_str:
            coords = [float(c) for c in cleaned_str.split()]
        else:
            coords = [float(c) for c in cleaned_str.split(',')]
        result = coords
        return result
    except (ValueError, SyntaxError) as e:
        return [-1, -1, -1, -1] # Return invalid bbox if parsing fails







def interpolate_sequence(frames, values, max_width=None, max_height=None):
    """Interpolate values over given frame numbers, ignoring invalid bboxes (-1)."""
    frames = np.array(frames)
    values = np.array(values, dtype=float)

    # Replace invalid bbox (-1) with np.nan for interpolation
    values[values < 0] = np.nan

    all_frames = np.arange(frames[0], frames[-1] + 1)

    # Interpolate each coordinate separately
    interpolated_values = np.zeros((len(all_frames), 4))
    for i in range(4):
        coord = values[:, i]
        valid_mask = ~np.isnan(coord)

        # If no valid values for this coordinate, fill with NaNs
        if valid_mask.sum() == 0:
            interpolated_values[:, i] = np.nan
        elif valid_mask.sum() == 1:
            # If only one valid value, propagate it across all frames
            interpolated_values[:, i] = coord[valid_mask][0]
        else:
            interp_func = interp1d(
                frames[valid_mask], coord[valid_mask],
                kind='linear', bounds_error=False, fill_value="extrapolate"
            )
            interpolated_values[:, i] = interp_func(all_frames)

    # Clamp coordinates to [0, max_width/max_height]
    if max_width is not None and max_height is not None:
        for i, bbox in enumerate(interpolated_values):
            if not np.isnan(bbox).any():
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, max_width))
                x2 = max(0, min(x2, max_width))
                y1 = max(0, min(y1, max_width))
                y2 = max(0, min(y2, max_height))
                interpolated_values[i] = [x1, y1, x2, y2]

    return all_frames, interpolated_values


def interpolate_bounding_boxes(data, video_path=None):
    # Try to detect frame dimensions from video
    frame_width, frame_height = None, None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    # If no video, infer max width/height from data
    if frame_width is None or frame_height is None:
        max_x, max_y = 0, 0
        for index, row in data.iterrows(): # Iterate using iterrows for pandas DataFrame
            try:
                car_bbox = parse_bbox(row['car_bbox'])
                lp_bbox = parse_bbox(row['license_plate_bbox'])
                bboxes = [car_bbox, lp_bbox]
                for bbox in bboxes:
                    if bbox and all(v >= 0 for v in bbox):
                        x1, y1, x2, y2 = bbox
                        max_x = max(max_x, x1, x2)
                        max_y = max(max_y, y1, y2)
            except Exception:
                continue
        frame_width, frame_height = int(max_x), int(max_y)

    # --- Start of new logic for unique car_id assignment ---
    license_to_car_id_map = {}
    next_car_id = 0

    # First pass: Identify unique license numbers with high scores and assign unique car_ids
    for index, row in data.iterrows():
        score = float(row.get('license_number_score', 0))
        raw_license_number = str(row.get('license_number', '')).strip()

        reformatted_license_number = ""
        if raw_license_number:
            cleaned_license_number = "".join(raw_license_number.split()).upper()
            match = re.match(r"([A-Z])([A-Z]{2})(\d{4})", cleaned_license_number)
            if match:
                letter1, letters23, digits4 = match.groups()
                reformatted_license_number = f"{letter1} {letters23} {digits4}"
        
        if score > 0.70 and reformatted_license_number:
            if reformatted_license_number not in license_to_car_id_map:
                license_to_car_id_map[reformatted_license_number] = next_car_id
                next_car_id += 1
            # Update the car_id in the DataFrame for this row
            data.loc[index, 'car_id'] = license_to_car_id_map[reformatted_license_number]
        else:
            # If score is low or license number is invalid, set car_id to -1 for now
            # This will be handled by the existing grouping logic later
            data.loc[index, 'car_id'] = -1
    # --- End of new logic ---

    # Group rows by car_id, skipping -1
    grouped = defaultdict(list)
    for index, row in data.iterrows():
        try:
            car_id = int(float(row['car_id']))
        except ValueError:
            continue
        if car_id == -1:  # skip invalid detections
            continue
        grouped[car_id].append(row)

    interpolated_data = []

    for car_id, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r['frame_number']))
        frames = [int(r['frame_number']) for r in rows]
        car_bboxes = [parse_bbox(row['car_bbox']) for row in rows]
        lp_bboxes = [parse_bbox(row['license_plate_bbox']) for row in rows]

        # Collect license numbers and scores for interpolation
        license_numbers_raw = []
        license_number_scores_raw = []
        license_plate_bbox_scores_raw = []
        for r in rows:
            score = float(r.get('license_number_score', 0))
            raw_license_number = str(r.get('license_number', '')).strip() # Ensure it's a string and strip whitespace
            
            # --- New string processing logic ---
            reformatted_license_number = ""
            if raw_license_number: # Only process if not empty
                cleaned_license_number = "".join(raw_license_number.split()).upper() # Strip all spaces and convert to uppercase
                
                # Regex to extract components: 1 letter, 2 letters, 4 digits
                match = re.match(r"([A-Z])([A-Z]{2})(\d{4})", cleaned_license_number)
                
                if match:
                    letter1, letters23, digits4 = match.groups()
                    reformatted_license_number = f"{letter1} {letters23} {digits4}"
            # --- End new string processing logic ---

            if score > 0.70 and reformatted_license_number: # Only use if score is high and reformatting was successful
                license_numbers_raw.append(reformatted_license_number)
                license_number_scores_raw.append(score)
                license_plate_bbox_scores_raw.append(float(r.get('license_plate_bbox_score', 0)))
            else:
                license_numbers_raw.append(np.nan) # Use NaN for low-confidence, missing, or non-conforming
                license_number_scores_raw.append(np.nan)
                license_plate_bbox_scores_raw.append(np.nan)

        # Convert to pandas Series for ffill/bfill
        ln_series = pd.Series(license_numbers_raw, index=frames)
        ls_series = pd.Series(license_number_scores_raw, index=frames)
        lp_bbox_s_series = pd.Series(index=frames, dtype=float) # Initialize with float dtype
        lp_bbox_s_series.loc[frames] = license_plate_bbox_scores_raw


        # Forward-fill and then backward-fill to propagate valid license numbers
        ln_series = ln_series.ffill().bfill()
        ls_series = ls_series.ffill().bfill()
        lp_bbox_s_series = lp_bbox_s_series.ffill().bfill()

        # If after ffill/bfill, all license numbers are still NaN, then skip this car_id
        if ln_series.isnull().all():
            continue # Skip this car_id if no valid license number was ever found

        # Interpolate both car & license plate bboxes with clamping
        all_frames, car_bboxes_interp = interpolate_sequence(frames, car_bboxes, frame_width, frame_height)
        _, lp_bboxes_interp = interpolate_sequence(frames, lp_bboxes, frame_width, frame_height)

        # Create a full series for license numbers and scores based on all_frames
        full_ln_series = pd.Series(index=all_frames, dtype=object)
        full_ls_series = pd.Series(index=all_frames, dtype=float)
        full_lp_bbox_s_series = pd.Series(index=all_frames, dtype=float)
        
        # Fill with known values
        full_ln_series.loc[frames] = ln_series.values
        full_ls_series.loc[frames] = ls_series.values
        full_lp_bbox_s_series.loc[frames] = lp_bbox_s_series.values

        # Propagate again over the full frame range
        full_ln_series = full_ln_series.ffill().bfill()
        full_ls_series = full_ls_series.ffill().bfill()
        full_lp_bbox_s_series = full_lp_bbox_s_series.ffill().bfill()

        for i, f in enumerate(all_frames):
            # Replace np.nan with -1 in interpolated bboxes before converting to string
            car_bbox_processed = [float(x) if not np.isnan(x) else -1.0 for x in car_bboxes_interp[i]]
            lp_bbox_processed = [float(x) if not np.isnan(x) else -1.0 for x in lp_bboxes_interp[i]]

            row = {
                'frame_number': f,
                'car_id': car_id,
                'car_bbox': str(car_bbox_processed),
                'license_plate_bbox': str(lp_bbox_processed),
                'license_number': full_ln_series.get(f, '0'), # Use propagated value
                'license_number_score': full_ls_series.get(f, 0.0), # Use propagated value
                'license_plate_bbox_score': full_lp_bbox_s_series.get(f, 0.0) # This score is for the LP detection, not OCR
            }
            interpolated_data.append(row)

    return interpolated_data
