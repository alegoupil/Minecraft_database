import cv2
import os

# Paths
IMAGE_DIR = "./train/images/"
LABEL_DIR = "./train/labels/"
CLASSES_PATH = "./classes.txt"
start_index = 0

# Load class names from classes.txt
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    print("Warning: classes.txt not found. Defaulting to class IDs.")
    class_names = []

# Get sorted list of images
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])

print("------ YOLO Image Viewer ------")
print(f"Total images found: {len(image_files)}\n")

index = start_index

while 0 <= index < len(image_files):
    filename = image_files[index]
    image_path = os.path.join(IMAGE_DIR, filename)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {filename}")
        index += 1
        continue

    height, width = image.shape[:2]

    # Read YOLO label file if it exists
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)

                # Convert normalized values to pixel values
                x_center *= width
                y_center *= height
                w *= width
                h *= height

                x_min = int(x_center - w / 2)
                y_min = int(y_center - h / 2)
                x_max = int(x_center + w / 2)
                y_max = int(y_center + h / 2)

                # Get label name from classes.txt if available
                label = class_names[int(class_id)] if int(class_id) < len(class_names) else f"Class {int(class_id)}"

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, label, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show image with bounding boxes
    cv2.imshow("YOLO Labeled Image", image)
    cv2.setWindowTitle("YOLO Labeled Image", filename)
    key = cv2.waitKey(0)

    if key == 27:  # Esc
        break
    elif key == ord('d'):  # Next image
        index += 1
    elif key == ord('q'):  # Previous image
        index -= 1

cv2.destroyAllWindows()
