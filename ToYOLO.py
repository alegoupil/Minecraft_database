import cv2
import pandas as pd
import os
import shutil

# Path to the CSV file and image directory
CSV_PATH = "data.csv"
IMAGE_DIR = "./Images/"

output_folder = IMAGE_DIR + "train/"

# Read the CSV file
df = pd.read_csv(CSV_PATH)

print("CSV imported")

if not os.path.exists(output_folder):
     os.makedirs(output_folder)
index = 0
old_image = None
class_id = {
    "SPIDER" : 0,
    "SLIME" : 1,
    "WITCH" : 2,
    "PIG" : 3,
    "ZOMBIE" : 4,
    "SKELETON" : 5,
    "ENDERMAN" : 6,
    "CREEPER" : 7
}

# Group bounding boxes by image
image_annotations = {}
for _, row in df.iterrows():
    filename = row["Image"]
    x_min, y_min, width, height = int(row["xMin"]), int(row["yMin"]), int(row["width"]), int(row["height"])
    x_center, y_center = x_min + width/2, y_min + height/2
    label = row["MobType"]
    
    if filename != old_image:
        old_image = filename
        index += 1
        shutil.copy(IMAGE_DIR + filename, output_folder + str(index) + ".png")

    label_path = f"{output_folder}/{index}.txt"
    with open(label_path, "a") as f:
        f.write(f"{class_id[label]} {x_center} {y_center} {width} {height}\n")