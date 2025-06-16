import string
import easyocr
import numpy as np
import re

# note: plate detection model got usable in train-4
# and recognition test got usable data from test3.csv


from sklearn.cluster import DBSCAN # for density based clustering

# Initialize the OCR reader in nepali and english
reader = easyocr.Reader(['ne','en', 'hi'], gpu=True)



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


def read_license_plate(license_plate_grayscale, reader):
    """ process grayscaled license plate based on line configuration """
    
    ocr_results = reader.readtext(license_plate_grayscale)
    
    if not ocr_results:
        return "", 0.0  # Return empty text and no confidence if OCR fails
    
    # average confidence score
    avg_conf = np.mean([conf for _, _, conf in ocr_results]) if ocr_results else 0.0
    
    
    # determine plate type
    plate_type = detect_plate_lines(license_plate_grayscale, ocr_results)
    
    # handle different plate types
    if plate_type == "single_line":
        plate_text = " ".join([text for _, text, _ in ocr_results])
        plate_text = normalize_text(plate_text) # normalize plate text
        # avg_conf = np.mean([conf for _, _, conf in ocr_results]) if ocr_results else 0.0
        
    
    elif plate_type == "multi_line":
        # Get character heights for smart splitting
        char_heights = [box[2][1] - box[0][1] for box, _, _ in ocr_results]
        avg_height = np.mean(char_heights) if char_heights else 0
        
        # Find optimal split point
        y_centers = [(box[0][1] + box[2][1]) / 2 for box, _, _ in ocr_results]
        optimal_split = np.median(y_centers) if y_centers else license_plate_grayscale.shape[0] / 2
        
        # Initialize lists
        upper_text = []
        lower_text = []
        middle_chars = []   # stores (box, text) tuples
        
        # Classify characters based on position
        for box, text, _ in ocr_results:
            center_y = (box[0][1] + box[2][1]) / 2
            threshold = avg_height * 0.2
            
            if center_y < optimal_split - threshold:
                upper_text.append(text)
            elif center_y > optimal_split + threshold:
                lower_text.append(text)
            else:
                middle_chars.append((box, text))
        
        # Distribute middle characters based on proximity
        for box, text in middle_chars:
            center_y = (box[0][1] + box[2][1]) / 2
            upper_dist = abs(center_y - (optimal_split - threshold))
            lower_dist = abs(center_y - (optimal_split + threshold))
            
            if upper_dist < lower_dist:
                upper_text.append(text)
            else:
                lower_text.append(text)
                
        plate_text =  " ".join(upper_text) + " " + " ".join(lower_text)
        plate_text = normalize_text(plate_text) # normalize plate text
        # avg_conf = np.mean([conf for _, _, conf in ocr_results]) if ocr_results else 0.0
        
    else:
        # fallback
        plate_text = " ".join([text for _, text, _ in ocr_results])
        plate_text = normalize_text(plate_text) # normalize plate text
        # avg_conf = np.mean([conf for _, _, conf in ocr_results]) if ocr_results else 0.0
        
        
        
    if is_english(plate_text):
        if complies_embosed_format(plate_text):
            plate_text = correct_embosed_plate(plate_text)
            return plate_text, avg_conf
    
    elif is_nepali(plate_text):
        if match_old_format(plate_text):
            return plate_text, avg_conf
        elif match_province_format(plate_text):
            return plate_text, avg_conf
        # else:
        #     # If it doesn't match any known format, return as is
        #     return plate_text, avg_conf

    return plate_text, avg_conf
    # return "", 0.0


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




def match_old_format(text):
    """
    -- for Old Nepali plate format --
    Matches plates like: 
    बा १ च १२३४
    """
    pattern = r"[\u0900-\u097F]{1,2}\s*[\d\u0966-\u096F]{1,2}\s*[\u0900-\u097F]{1,2}\s*[\d\u0966-\u096F]{1,4}"

    return bool(re.search(pattern, text))


def match_province_format(text):
    """
    -- for New Province format --
    Matches multi-line Nepali plate after joining lines.
    Example normalized text: 
    'प्रदेश ३-०२ ००१-च १२३४'
    """
    pattern = r"(?:[\u0900-\u097F]+\s*)?प्रदेश\s*[\d\u0966-\u096F]{1}-[\d\u0966-\u096F]{1,2}\s+[\d\u0966-\u096F]{1,3}[-‐][\u0900-\u097F]{1,2}\s+[\d\u0966-\u096F]{1,4}"


    return bool(re.search(pattern, text))


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

def is_nepali(text):
    # Returns True if text contains any Devanagari character (Nepali script)
    for char in text:
        if '\u0900' <= char <= '\u097F':
            return True
    return False







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
                print(results[frame_number][car_id])
                if 'car' in results[frame_number][car_id].keys() and \
                   'license_plate' in results[frame_number][car_id].keys() and \
                   'text' in results[frame_number][car_id]['license_plate'].keys():
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
        f.close()



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