# import ast

# import cv2
# import numpy as np
# import pandas as pd


# def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
#     x1, y1 = top_left
#     x2, y2 = bottom_right

#     cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
#     cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

#     cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
#     cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

#     cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
#     cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

#     cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
#     cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

#     return img

# interpolated_file_path = 'interpolated_results/test_interpolated.csv'
# results = pd.read_csv(interpolated_file_path)


# """

# # Load camera (0 = default webcam)
# cap = cv2.VideoCapture(0)

# # Check if camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # TODO: Pass `frame` to your license plate detection function here
#     # e.g. detect_license_plate(frame)

#     # Show live feed (optional for visualization)
#     cv2.imshow("Camera Feed", frame)

#     # Exit on 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release camera and close windows
# cap.release()
# cv2.destroyAllWindows()

# """


# # load video
# video_path = './videos/plate_test.mp4'
# cap = cv2.VideoCapture(video_path)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('./output/out.mp4', fourcc, fps, (width, height))

# license_plate = {}
# for car_id in np.unique(results['car_id']):
#     max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
#     license_plate[car_id] = {'license_crop': None,
#                              'license_plate_number': results[(results['car_id'] == car_id) &
#                                                              (results['license_number_score'] == max_)]['license_number'].iloc[0]}
#     cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
#                                              (results['license_number_score'] == max_)]['frame_number'].iloc[0])
#     ret, frame = cap.read()

#     x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
#                                               (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

#     license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#     license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

#     license_plate[car_id]['license_crop'] = license_crop


# frame_number = -1

# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# # read frames
# ret = True
# while ret:
#     ret, frame = cap.read()
#     frame_number += 1
#     if ret:
#         df_ = results[results['frame_number'] == frame_number]
#         for row_indx in range(len(df_)):
#             # draw car
#             car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
#             draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
#                         line_length_x=200, line_length_y=200)

#             # draw license plate
#             x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

#             # crop license plate
#             license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

#             H, W, _ = license_crop.shape

#             try:
#                 frame[int(car_y1) - H - 100:int(car_y1) - 100,
#                       int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

#                 frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
#                       int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

#                 (text_width, text_height), _ = cv2.getTextSize(
#                     license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     4.3,
#                     17)

#                 cv2.putText(frame,
#                             license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
#                             (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             4.3,
#                             (0, 0, 0),
#                             17)

#             except:
#                 pass

#         out.write(frame)
#         frame = cv2.resize(frame, (1280, 720))

#         # Show the frame
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


# out.release()
# cap.release()


###############################
import ast
import cv2
import numpy as np
import pandas as pd
import os
import argparse # Added argparse

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Added argument parsing
parser = argparse.ArgumentParser(description="Visualize license plate tracking results.")
parser.add_argument('--interpolated_csv', type=str, required=True,
                    help='Path to the interpolated CSV file containing tracking results.')
parser.add_argument('--video_path', type=str, required=True,
                    help='Path to the original video file.')
parser.add_argument('--output_video_path', type=str, default='./output/out.mp4',
                    help='Path to save the output visualized video.')
args = parser.parse_args()

video_path = args.video_path # Re-inserted this line

def parse_bbox_string(bbox_str):
    try:
        # Remove brackets and split by comma
        parts = bbox_str.strip('[]').split(',')
        # Convert each part to float, handling 'nan'
        parsed_bbox = []
        for p in parts:
            p_stripped = p.strip()
            if p_stripped == 'nan':
                parsed_bbox.append(np.nan)
            elif p_stripped: # Avoid empty strings from split
                parsed_bbox.append(float(p_stripped))
        return parsed_bbox
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing bbox string '{bbox_str}': {e}. Returning [np.nan, np.nan, np.nan, np.nan]")
        return [np.nan, np.nan, np.nan, np.nan] # Return a default safe bbox

interpolated_file_path = args.interpolated_csv
# Use converters to apply the custom parsing function to the bbox columns
results = pd.read_csv(interpolated_file_path, converters={
    'car_bbox': parse_bbox_string,
    'license_plate_bbox': parse_bbox_string
})
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs(os.path.dirname(args.output_video_path), exist_ok=True) # Ensure output directory exists

out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': results[(results['car_id'] == car_id) &
                                        (results['license_number_score'] == max_)]['license_number'].iloc[0]
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_number'].iloc[0])
    ret, frame = cap.read()

    if not ret:
        continue

    x1, y1, x2, y2 = results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0]

    # Clip values to frame boundaries
    # Check for NaN values before converting to int
    if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))

        license_plate[car_id]['license_crop'] = frame[y1:y2, x1:x2, :]
        license_plate[car_id]['bbox'] = [x1, y1, x2, y2]

frame_number = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret = True
while ret:
    ret, frame = cap.read()
    frame_number += 1
    if ret:
        df_ = results[results['frame_number'] == frame_number]
        for row_indx in range(len(df_)):
            try:
                car_x1, car_y1, car_x2, car_y2 = df_.iloc[row_indx]['car_bbox']
                if any(np.isnan([car_x1, car_y1, car_x2, car_y2])):
                    continue
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                x1, y1, x2, y2 = df_.iloc[row_indx]['license_plate_bbox']
                if any(np.isnan([x1, y1, x2, y2])):
                    continue
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
                if license_crop is None:
                    continue

                H, W, _ = license_crop.shape

                # Safe placement coordinates
                top_y = max(0, int(car_y1) - H - 100)
                bottom_y = top_y + H
                left_x = max(0, int((car_x2 + car_x1 - W) / 2))
                right_x = left_x + W

                # Check frame boundary
                if bottom_y <= frame.shape[0] and right_x <= frame.shape[1]:
                    frame[top_y:bottom_y, left_x:right_x, :] = license_crop

                    # White rectangle for text
                    rect_top = max(0, top_y - 300)
                    rect_bottom = top_y
                    if rect_bottom > rect_top:
                        frame[rect_top:rect_bottom, left_x:right_x, :] = (255, 255, 255)

                        (text_width, text_height), _ = cv2.getTextSize(
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)

                        text_x = max(0, int((car_x2 + car_x1 - text_width) / 2))
                        text_y = rect_top + (rect_bottom - rect_top) // 2 + text_height // 2

                        cv2.putText(frame,
                                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)

            except Exception as e:
                print(f"[ERROR] Frame {frame_number}, Car ID {df_.iloc[row_indx]['car_id']} — {str(e)}")

        out.write(frame)
        # try:
        #     frame_resized = cv2.resize(frame, (1280, 720))
        #     cv2.imshow('frame', frame_resized)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # except cv2.error:
        #     # Skip if GUI not supported
        #     pass

out.release()
cap.release()
# cv2.destroyAllWindows()
