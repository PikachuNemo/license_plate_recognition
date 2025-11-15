import string
import easyocr
import numpy as np
import re
import pandas as pd
from collections import defaultdict

# note: plate detection model got usable in train-4
# and recognition test got usable data from test3.csv


from sklearn.cluster import DBSCAN # for density based clustering



def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_car(license_plate, vehicle_track_ids):
    """
    Assigns a license plate to a vehicle based on the Intersection over Union (IoU).

    Args:
        license_plate (list): A list containing the bounding box of the license plate [x1, y1, x2, y2].
        vehicle_track_ids (list): A list of vehicle track IDs.

    Returns:
        tuple: A tuple containing the bounding box and ID of the assigned vehicle.
    """
    x1, y1, x2, y2 = license_plate

    max_iou = 0
    max_car_id = -1
    max_car_bbox = [-1, -1, -1, -1] # Initialize with default invalid values

    for car_track in vehicle_track_ids:
        car_x1, car_y1, car_x2, car_y2, car_id = car_track
        iou = bb_intersection_over_union([x1, y1, x2, y2], [car_x1, car_y1, car_x2, car_y2])
        if iou > max_iou:
            max_iou = iou
            max_car_id = car_id
            max_car_bbox = [car_x1, car_y1, car_x2, car_y2]

    return max_car_bbox, max_car_id


def read_license_plate(license_plate_image, reader):
    """ process pre-processed license plate image and return the raw text """

    # Use paragraph=False to get individual confidence scores.
    ocr_results = reader.readtext(license_plate_image, paragraph=False)



    if not ocr_results:
        return "", 0.0  # Return empty text and no confidence if OCR fails

    # Join text and calculate average confidence
    plate_text = " ".join([result[1] for result in ocr_results])
    confidence = np.mean([result[2] for result in ocr_results])

    plate_text = normalize_text(plate_text)  # normalize plate text

    # Filter for English plates only
    if not is_english(plate_text):
        return "", 0.0 # Return empty if not an English plate

    return plate_text, confidence


# def read_license_plate(license_plate_crop):
#     """
    
#     """
#     detections = reader.readtext(license_plate_crop)

#     for detection in detections:
#         bbox, text, conf_score = detection

#         text = text.upper().replace(' ', '')
        
#         return text, conf_score

#         # if license_complies_format(text):
#         #     return format_license(text), conf_score

#     return None, None



def detect_plate_lines(license_plate_crop, results):
    """
    Determine if license plate is single or multi-line
    Returns: "single_line", "multi_line", or "unknown"
    """
    # Handle empty results
    if not results:
        return "unknown"
    
    plate_height = license_plate_crop.shape[0]
    y_centers = np.array([(box[0][1] + box[2][1]) / 2 for box, _, _ in results])
    
    # Calculate normalized spread (0-1 scale)
    spread = (np.max(y_centers) - np.min(y_centers)) / plate_height
    
    # Case 1: Small spread = single line
    if spread < 0.15:  # Threshold 1: 15% of plate height
        return "single_line"
    
    # Case 2: Use clustering to detect distinct lines
    clustering = DBSCAN(eps=0.1*plate_height, min_samples=1).fit(y_centers.reshape(-1, 1))
    unique_labels = set(clustering.labels_)
    
    # Case 3: Check if clusters are vertically separated
    cluster_centers = [np.mean(y_centers[clustering.labels_ == label]) for label in unique_labels]
    cluster_centers.sort()
    
    # Calculate vertical gaps between clusters
    gaps = [cluster_centers[i+1] - cluster_centers[i] for i in range(len(cluster_centers)-1)]
    
    # If significant gap exists between clusters
    if any(gap > 0.15 * plate_height for gap in gaps) and len(unique_labels) > 1:
        return "multi_line"
    
    # Case 4: Fallback to spread threshold
    return "multi_line" if spread > 0.25 else "single_line"  # Threshold 2: 25% height






# normalizing/standardizing OCR_results
def normalize_text(text):
    """Fix OCR-related noise in recognized text."""
    text = text.replace("‐", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text.strip())  # collapse multiple spaces to one
    return text




# ------------- license plate format validation for Embosed number plates, older nepali format and province format -------------- #

def correct_embosed_plate(text):
    """
    -- for Embosed number plate --
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: None, 2: dict_int_to_char, 3: dict_int_to_char, 4: None,
               5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int}
    
    for j in range(9):
        if j in mapping:
            map_entry = mapping[j]
            if map_entry and text[j] in map_entry:
                license_plate_ += map_entry[text[j]]
            else:
                license_plate_ += text[j]
        else:
            license_plate_ += text[j]

    return license_plate_

def complies_embosed_format(text):
    """
    -- for English plate format --
    Match license plate format like 'B AE 5627'
    
    Args:
        text (str): License plate text.
    Returns:
        bool: True if the license plate text matches the format, False otherwise.
    """
    if len(text) != 9:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
        (text[1] == ' ') and \
        (text[2] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
        (text[3] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
        (text[4] == ' ') and \
        (text[5] in string.digits or text[3] in dict_char_to_int.keys()) and \
        (text[6] in string.digits or text[4] in dict_char_to_int.keys()) and \
        (text[7] in string.digits or text[5] in dict_char_to_int.keys()) and \
        (text[8] in string.digits or text[6] in dict_char_to_int.keys()):
        return True
    else:
        return False




# def match_old_format(text):
#     """
#     -- for Old Nepali plate format --
#     Matches plates like: 
#     बा १ च १२३४
#     """
#     pattern = r"[\u0900-\u097F]{1,2}\s*[\d\u0966-\u096F]{1,2}\s*[\u0900-\u097F]{1,2}\s*[\d\u0966-\u096F]{1,4}"

#     return bool(re.search(pattern, text))


# def match_province_format(text):
#     """
#     -- for New Province format --
#     Matches multi-line Nepali plate after joining lines.
#     Example normalized text: 
#     'प्रदेश ३-०२ ००१-च १२३४'
#     """
#     pattern = r"(?:[\u0900-\u097F]+\s*)?प्रदेश\s*[\d\u0966-\u096F]{1}-[\d\u0966-\u096F]{1,2}\s+[\d\u0966-\u096F]{1,3}[-‐][\u0900-\u097F]{1,2}\s+[\d\u0966-\u096F]{1,4}"


#     return bool(re.search(pattern, text))


def is_english(text):
    """ Check if the text contains only English uppercase letters, digits, and spaces.
    args:
        text (str): The input text to check.
    returns:
        boolean value
    """
    return all(
        char.isascii() and (char.isupper() or char.isdigit() or char.isspace())
        for char in text
    )

# def is_nepali(text):
#     # Returns True if text contains any Devanagari character (Nepali script)
#     for char in text:
#         if '\u0900' <= char <= '\u097F':
#             return True
#     return False







### -------------------------------- ###




def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_number in results.keys():
            for car_id in results[frame_number].keys():
                if 'car' in results[frame_number][car_id].keys() and \
                   'license_plate' in results[frame_number][car_id].keys() and \
                   'text' in results[frame_number][car_id]['license_plate'].keys():
                    try:
                        f.write('{},{},{},{},{},{},{}\n'.format(frame_number,
                                                                car_id,
                                                                '[{} {} {} {}]'.format(
                                                                    results[frame_number][car_id]['car']['bbox'][0],
                                                                    results[frame_number][car_id]['car']['bbox'][1],
                                                                    results[frame_number][car_id]['car']['bbox'][2],
                                                                    results[frame_number][car_id]['car']['bbox'][3]),
                                                                 '[{} {} {} {}]'.format(
                                                                    results[frame_number][car_id]['license_plate']['bbox'][0],
                                                                    results[frame_number][car_id]['license_plate']['bbox'][1],
                                                                    results[frame_number][car_id]['license_plate']['bbox'][2],
                                                                    results[frame_number][car_id]['license_plate']['bbox'][3]),
                                                                results[frame_number][car_id]['license_plate']['bbox_score'],
                                                                results[frame_number][car_id]['license_plate']['text'],
                                                                results[frame_number][car_id]['license_plate']['text_score'])
                                )
                    except Exception as e:
                        print(f"[write_csv] Error writing row for frame {frame_number}, car_id {car_id}: {e}")
                        raise



# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'D': '0',
                    'Q': '0',
                    'I': '1',
                    'Z': '2',
                    'J': '3',
                    'A': '4',
                    'S': '5',
                    'G': '6',
                    'B': '8'
                    }

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '2': 'Z',
                    '3': 'J',
                    '4': 'A',
                    '5': 'S',
                    '6': 'G',
                    '8': 'B',
                    '9': 'P'
                    }


def interpolate_license_plates(raw_plate_data_list):
    """
    Interpolates missing license plate readings for each car_id across frames.
    If multiple readings exist for a car_id in a frame, the one with the highest
    confidence score is chosen. Missing readings are filled using forward fill,
    then backward fill.

    Args:
        raw_plate_data_list (list): A list of dictionaries, where each dictionary
                                     represents a raw license plate detection.

    Returns:
        list: A list of dictionaries with interpolated license plate data.
    """
    if not raw_plate_data_list:
        return []

    df = pd.DataFrame(raw_plate_data_list)

    # Ensure 'frame_number' and 'car_id' are suitable for grouping and sorting
    df['frame_number'] = df['frame_number'].astype(int)
    df['car_id'] = df['car_id'].astype(int)

    interpolated_results = []

    # Group by car_id to process each car's trajectory independently
    for car_id, group in df.groupby('car_id'):
        # Sort by frame number to ensure correct interpolation order
        group = group.sort_values(by='frame_number')

        # For frames with multiple detections for the same car_id,
        # select the one with the highest license_number_score
        idx = group.groupby('frame_number')['license_number_score'].idxmax()
        group = group.loc[idx]

        # Reindex to fill in missing frames for interpolation
        # Create a complete range of frames for this car's presence
        min_frame = group['frame_number'].min()
        max_frame = group['frame_number'].max()
        full_frame_range = pd.DataFrame({'frame_number': range(min_frame, max_frame + 1)})

        # Merge with the group data, filling missing frames with NaNs
        # Use 'outer' merge to keep all frames from full_frame_range
        merged_group = pd.merge(full_frame_range, group, on='frame_number', how='left')

        # Interpolate 'license_number' and 'license_number_score'
        # For license_number, we'll use forward fill then backward fill
        merged_group['license_number'] = merged_group['license_number'].ffill().bfill()
        merged_group['license_number_score'] = merged_group['license_number_score'].ffill().bfill()

        # Fill other missing values (like bboxes) with the nearest available
        # This is a simple approach; more sophisticated bbox interpolation might be needed
        # For simplicity, we'll ffill and bfill other relevant columns
        merged_group['car_id'] = merged_group['car_id'].ffill().bfill()
        merged_group['car_bbox'] = merged_group['car_bbox'].ffill().bfill()
        merged_group['license_plate_bbox'] = merged_group['license_plate_bbox'].ffill().bfill()
        merged_group['license_plate_bbox_score'] = merged_group['license_plate_bbox_score'].ffill().bfill()


        # After interpolation, convert back to list of dicts and append
        interpolated_results.extend(merged_group.to_dict(orient='records'))
    
    # Sort the final results by frame_number and then car_id
    interpolated_results_df = pd.DataFrame(interpolated_results)
    if not interpolated_results_df.empty:
        interpolated_results_df = interpolated_results_df.sort_values(by=['frame_number', 'car_id']).reset_index(drop=True)
        return interpolated_results_df.to_dict(orient='records')
    else:
        return []


def scale_bbox(x1, y1, x2, y2, original_width, original_height, resized_width, resized_height):
    """
    Scales bounding box coordinates from a resized frame back to the original frame dimensions.

    Args:
        x1, y1, x2, y2 (float): Bounding box coordinates in the resized frame.
        original_width (int): Width of the original frame.
        original_height (int): Height of the original frame.
        resized_width (int): Width of the resized frame.
        resized_height (int): Height of the resized frame.

    Returns:
        tuple: Scaled bounding box coordinates (x1_scaled, y1_scaled, x2_scaled, y2_scaled).
    """
    x1_scaled = int(x1 * (original_width / resized_width))
    y1_scaled = int(y1 * (original_height / resized_height))
    x2_scaled = int(x2 * (original_width / resized_width))
    y2_scaled = int(y2 * (original_height / resized_height))
    return x1_scaled, y1_scaled, x2_scaled, y2_scaled
