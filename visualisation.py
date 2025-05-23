import cv2
import pandas as pd
import os

# Path to the CSV file and image directory
CSV_PATH = "data_reduc.csv"
IMAGE_DIR = "./images_reduc/"
start_index = 0  # Change this value to start at a different image

# Read the CSV file
df = pd.read_csv(CSV_PATH)

print("CSV imported")

total_classifications = 0

# Group bounding boxes by image
image_annotations = {}
for _, row in df.iterrows():
    filename = row["Image"]
    x_min, y_min, width, height = int(row["xMin"]), int(row["yMin"]), int(row["width"]), int(row["height"])
    x_max, y_max = x_min + width, y_min + height
    label = row["MobType"]
    
    if filename not in image_annotations:
        image_annotations[filename] = []
    image_annotations[filename].append((x_min, y_min, x_max, y_max, label))

    total_classifications += 1

print("Data linked to images")

# Get sorted list of images
image_files = sorted(os.listdir(IMAGE_DIR))

print("\n------ Results ------")
print(f"Total images in folder : {len(image_files)}")
print(f"Images with presence : {len(image_annotations)}")
print(f"Total classifications : {total_classifications}")
print("")

# Start index (modify as needed)
if start_index < 0 or start_index >= len(image_files):
    start_index = 0

index = start_index

while 0 <= index < len(image_files):
    filename = image_files[index]
    image_path = os.path.join(IMAGE_DIR, filename)
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        index += 1
        continue
        
    # Draw bounding boxes
    if filename in image_annotations:
        for x_min, y_min, x_max, y_max, label in image_annotations[filename]:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Update the window with new image and title
    cv2.imshow("Labeled Image", image)
    cv2.setWindowTitle("Labeled Image", filename)
    key = cv2.waitKey(0)
        
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('d'):  # Press 'D' to go to the next image
        index += 1
    elif key == ord('q'):  # Press 'A' to go back to the previous image
        index -= 1

cv2.destroyAllWindows()
