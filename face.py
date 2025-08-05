import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_red_face(img_path, pad=100):
    img = cv2.imread(img_path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([160, 50, 70])
    upper_red1 = np.array([180, 180, 255])

    lower_red2 = np.array([0, 50, 70])
    upper_red2 = np.array([15, 180, 255])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)


    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)


    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])

    cropped = img[y1:y2, x1:x2]
    return cropped


def process_dataset(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)

            cropped = extract_red_face(input_path)
            if cropped is not None:

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, cropped)

input_dataset = "./test"
output_dataset = "./b"
process_dataset(input_dataset, output_dataset)
