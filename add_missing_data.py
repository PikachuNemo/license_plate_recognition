import csv
import numpy as np
from scipy.interpolate import interp1d
import os
from collections import defaultdict
import cv2
import pandas as pd


def parse_bbox(bbox_input):
    """Convert bbox string '[x1 y1 x2 y2]' or list to float list."""
    if isinstance(bbox_input, list):
        return list(map(float, bbox_input))
    if isinstance(bbox_input, str):
        return list(map(float, bbox_input.strip("[]").replace(",", " ").split()))
    return [] # Return empty list for other types


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

        # If no valid values for this coordinate, fill with zeros
        if valid_mask.sum() == 0:
            interpolated_values[:, i] = 0
        else:
            interp_func = interp1d(
                frames[valid_mask], coord[valid_mask],
                kind='linear', bounds_error=False, fill_value="extrapolate"
            )
            interpolated_values[:, i] = interp_func(all_frames)

    # Clamp coordinates to [0, max_width/max_height]
    if max_width is not None and max_height is not None:
        for i, bbox in enumerate(interpolated_values):
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, max_width))
            x2 = max(0, min(x2, max_width))
            y1 = max(0, min(y1, max_height))
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
        for row in data:
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

    # Group rows by car_id, skipping -1
    grouped = defaultdict(list)
    for row in data:
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
        car_bboxes = [parse_bbox(r['car_bbox']) or [-1, -1, -1, -1] for r in rows]
        lp_bboxes = [parse_bbox(r['license_plate_bbox']) or [-1, -1, -1, -1] for r in rows]

        # Collect license numbers and scores for interpolation
        license_numbers_raw = []
        license_number_scores_raw = []
        for r in rows:
            score = float(r.get('license_number_score', 0))
            if score > 0.70: # Apply threshold here
                license_numbers_raw.append(r.get('license_number', ''))
                license_number_scores_raw.append(score)
            else:
                license_numbers_raw.append(np.nan) # Use NaN for low-confidence or missing
                license_number_scores_raw.append(np.nan)

        # Convert to pandas Series for ffill/bfill
        ln_series = pd.Series(license_numbers_raw, index=frames)
        ls_series = pd.Series(license_number_scores_raw, index=frames)

        # Forward-fill and then backward-fill to propagate valid license numbers
        ln_series = ln_series.ffill().bfill()
        ls_series = ls_series.ffill().bfill()

        # If after ffill/bfill, all license numbers are still NaN, then skip this car_id
        if ln_series.isnull().all():
            continue # Skip this car_id if no valid license number was ever found

        # Interpolate both car & license plate bboxes with clamping
        all_frames, car_bboxes_interp = interpolate_sequence(frames, car_bboxes, frame_width, frame_height)
        _, lp_bboxes_interp = interpolate_sequence(frames, lp_bboxes, frame_width, frame_height)

        # Create a full series for license numbers and scores based on all_frames
        full_ln_series = pd.Series(index=all_frames, dtype=object)
        full_ls_series = pd.Series(index=all_frames, dtype=float)
        
        # Fill with known values
        full_ln_series.loc[frames] = ln_series.values
        full_ls_series.loc[frames] = ls_series.values

        # Propagate again over the full frame range
        full_ln_series = full_ln_series.ffill().bfill()
        full_ls_series = full_ls_series.ffill().bfill()

        for i, f in enumerate(all_frames):
            row = {
                'frame_number': str(f),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interp[i])),
                'license_plate_bbox': ' '.join(map(str, lp_bboxes_interp[i])),
                'license_number': str(full_ln_series.get(f, '0')), # Use propagated value
                'license_number_score': str(full_ls_series.get(f, '0.0')), # Use propagated value
                'license_plate_bbox_score': '0' # This score is for the LP detection, not OCR
            }
            interpolated_data.append(row)

    return interpolated_data
