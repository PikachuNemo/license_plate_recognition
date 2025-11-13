import string
import easyocr
import numpy as np
import re

# note: plate detection model got usable in train-4
# and recognition test got usable data from test3.csv


from sklearn.cluster import DBSCAN # for density based clustering



def get_car(license_plate, vehicle_track_ids):
    """
    
    """
    x1, y1, x2, y2, conf_score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_index = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_index]

    return -1, -1, -1, -1, -1
    # return None, None, None, None, -1


def read_license_plate(license_plate_image, reader):
    """ process pre-processed license plate image and return the raw text """

    # Use paragraph=True to handle multi-line plates. This returns a list of strings.
    ocr_results = reader.readtext(license_plate_image, paragraph=True)

    if not ocr_results:
        return "", 0.0  # Return empty text and no confidence if OCR fails

    # Join paragraphs (lines) with a newline character
    plate_text = "\n".join([result[1] for result in ocr_results])
    plate_text = normalize_text(plate_text)  # normalize plate text

    # Filter for English plates only
    if not is_english(plate_text):
        return "", 0.0 # Return empty if not an English plate

    # When paragraph=True, confidence scores are not available per block.
    # We'll return 1.0 as a placeholder confidence score indicating success.
    return plate_text, 1.0


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