import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

interpolated_file_path = 'results/interpolated_results/test_interpolated.csv'
results = pd.read_csv(interpolated_file_path)

video_path = 'data/videos/plate_test.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./output/out.mp4', fourcc, fps, (width, height))

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

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox']
                                      .iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop

frame_number = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret = True
while ret:
    ret, frame = cap.read()
    frame_number += 1
    if ret and frame is not None:
        df_ = results[results['frame_number'] == frame_number]
        for row_indx in range(len(df_)):
            try:
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox']
                                                                  .replace('[ ', '[').replace('   ', ' ')
                                                                  .replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox']
                                                  .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
                if license_crop is None:
                    continue

                H, W, _ = license_crop.shape

                y1_crop_start = max(0, int(car_y1) - H - 100)
                y1_crop_end = max(0, int(car_y1) - 100)

                crop_h = y1_crop_end - y1_crop_start
                if crop_h < H:
                    license_crop = license_crop[H - crop_h:]

                if y1_crop_end > y1_crop_start:
                    x_start = int((car_x2 + car_x1 - W) / 2)
                    x_end = int((car_x2 + car_x1 + W) / 2)

                    if y1_crop_end <= frame.shape[0] and x_end <= frame.shape[1]:
                        frame[y1_crop_start:y1_crop_end, x_start:x_end, :] = license_crop

                        y_text_start = max(0, y1_crop_start - 300)
                        frame[y_text_start:y1_crop_start, x_start:x_end, :] = (255, 255, 255)

                        (text_width, text_height), _ = cv2.getTextSize(
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)

                        cv2.putText(frame,
                                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                    (int((car_x2 + car_x1 - text_width) / 2), y_text_start + int(text_height / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
                    else:
                        print(f"[WARNING] Skipping license plate insertion at frame {frame_number} due to out-of-bounds.")
                else:
                    print(f"[WARNING] Skipping license plate insertion at frame {frame_number} due to invalid crop range.")

            except Exception as e:
                print(f"[ERROR] Frame {frame_number}, Car ID {df_.iloc[row_indx]['car_id']} — {str(e)}")

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # Skip GUI-based imshow if it fails
        try:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            print("[WARNING] Skipping frame display due to OpenCV GUI limitations.")
            break

out.release()
cap.release()
try:
    cv2.destroyAllWindows()
except:
    print("[WARNING] Unable to destroy OpenCV windows due to GUI limitations.")
