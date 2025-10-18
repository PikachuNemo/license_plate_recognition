# import csv
# import numpy as np
# from scipy.interpolate import interp1d
# import os


# def interpolate_bounding_boxes(data):
#     # Extract necessary data columns from input data
#     frame_numbers = np.array([int(row['frame_number']) for row in data])
#     car_ids = np.array([int(float(row['car_id'])) for row in data])
#     car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
#     license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

#     interpolated_data = []
#     unique_car_ids = np.unique(car_ids)
    
#     for car_id in unique_car_ids:
#         frame_numbers_ = [p['frame_number'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
#         print(frame_numbers_, car_id)

#         # Filter data for a specific car ID
#         car_mask = car_ids == car_id
#         car_frame_numbers = frame_numbers[car_mask]
#         car_bboxes_interpolated = []
#         license_plate_bboxes_interpolated = []

#         first_frame_number = car_frame_numbers[0]
#         last_frame_number = car_frame_numbers[-1]

#         for i in range(len(car_bboxes[car_mask])):
#             frame_number = car_frame_numbers[i]
#             car_bbox = car_bboxes[car_mask][i]
#             license_plate_bbox = license_plate_bboxes[car_mask][i]

#             if i > 0:
#                 prev_frame_number = car_frame_numbers[i-1]
#                 prev_car_bbox = car_bboxes_interpolated[-1]
#                 prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

#                 if frame_number - prev_frame_number > 1:
#                     # Interpolate missing frames' bounding boxes
#                     frames_gap = frame_number - prev_frame_number
#                     x = np.array([prev_frame_number, frame_number])
#                     x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
#                     interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
#                     interpolated_car_bboxes = interp_func(x_new)
#                     interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
#                     interpolated_license_plate_bboxes = interp_func(x_new)

#                     car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
#                     license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

#             car_bboxes_interpolated.append(car_bbox)
#             license_plate_bboxes_interpolated.append(license_plate_bbox)

#         for i in range(len(car_bboxes_interpolated)):
#             frame_number = first_frame_number + i
#             row = {}
#             row['frame_number'] = str(frame_number)
#             row['car_id'] = str(car_id)
#             row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
#             row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

#             if str(frame_number) not in frame_numbers_:
#                 # Imputed row, set the following fields to '0'
#                 row['license_plate_bbox_score'] = '0'
#                 row['license_number'] = '0'
#                 row['license_number_score'] = '0'
#             else:
#                 # Original row, retrieve values from the input data if available
#                 original_row = [p for p in data if int(p['frame_number']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
#                 row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
#                 row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
#                 row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

#             interpolated_data.append(row)

#     return interpolated_data

# # Input file path
# file_path = 'plate_recognition/recognition_test/test.csv'

# # Load the input file
# with open(file_path, 'r', encoding='utf-8') as file:
#     reader = csv.DictReader(file)
#     data = list(reader)

# # Interpolate missing data
# interpolated_data = interpolate_bounding_boxes(data)


# # Prepare output file path
# input_filename = os.path.basename(file_path)                          # test.csv
# filename_wo_ext, ext = os.path.splitext(input_filename)              # test, .csv
# output_filename = filename_wo_ext + '_interpolated' + ext            # test_interpolated.csv

# # Ensure output directory exists
# output_dir = 'interpolated_results'
# os.makedirs(output_dir, exist_ok=True)

# # Full output path
# output_path = os.path.join(output_dir, output_filename)


# # Write updated data to a new CSV file
# header = ['frame_number', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
# with open(output_path, 'w', newline='', encoding='utf-8') as file:
#     writer = csv.DictWriter(file, fieldnames=header)
#     writer.writeheader()
#     writer.writerows(interpolated_data)
    


### ----------------------------------------------



import csv
import numpy as np
from scipy.interpolate import interp1d
import os
from collections import defaultdict
import cv2


def parse_bbox(bbox_str: str):
    """Convert bbox string '[x1 x2 y1 y2]' to float list."""
    return list(map(float, bbox_str.strip("[]").replace(",", " ").split()))


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
        car_bboxes = [parse_bbox(r['car_bbox']) for r in rows]
        lp_bboxes = [parse_bbox(r['license_plate_bbox']) for r in rows]

        # Interpolate both car & license plate bboxes with clamping
        all_frames, car_bboxes_interp = interpolate_sequence(frames, car_bboxes, frame_width, frame_height)
        _, lp_bboxes_interp = interpolate_sequence(frames, lp_bboxes, frame_width, frame_height)

        frame_set = set(frames)
        for i, f in enumerate(all_frames):
            row = {
                'frame_number': str(f),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interp[i])),
                'license_plate_bbox': ' '.join(map(str, lp_bboxes_interp[i]))
            }

            if f not in frame_set:  # Interpolated frame
                row.update({
                    'license_plate_bbox_score': '0',
                    'license_number': '0',
                    'license_number_score': '0'
                })
            else:  # Original frame
                original = next(r for r in rows if int(r['frame_number']) == f)
                row.update({
                    'license_plate_bbox_score': original.get('license_plate_bbox_score', '0'),
                    'license_number': original.get('license_number', '0'),
                    'license_number_score': original.get('license_number_score', '0')
                })

            interpolated_data.append(row)

    return interpolated_data


# Input file path
file_path = 'plate_recognition/recognition_test/test.csv'

# Infer video path automatically from CSV base name
input_filename = os.path.basename(file_path)
filename_wo_ext, ext = os.path.splitext(input_filename)
video_dir = './videos'
video_path = os.path.join(video_dir, filename_wo_ext + '.mp4')

with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data with video dimensions or inferred values
interpolated_data = interpolate_bounding_boxes(data, video_path=video_path)

# Prepare output file path
output_filename = filename_wo_ext + '_interpolated' + ext
output_dir = 'interpolated_results'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_filename)

# Write updated data
header = ['frame_number', 'car_id', 'car_bbox', 'license_plate_bbox',
          'license_plate_bbox_score', 'license_number', 'license_number_score']
with open(output_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
