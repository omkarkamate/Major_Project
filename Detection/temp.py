import cv2
import os
import json


DATASET_PATH = "Detection/original_dataset"
REFERENCE_FOLDER = os.path.join(DATASET_PATH, "b1")

files = os.listdir(REFERENCE_FOLDER)

# Select the first image from b1
image_file = None

for file in files:
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_file = file
        break

if image_file is None:
    print("No image found in b1 folder.")
    exit()

image_path = os.path.join(
    REFERENCE_FOLDER,
    image_file
)

image = cv2.imread(image_path)
# cv2.imshow("Reference Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(image_file)
print(image_path)