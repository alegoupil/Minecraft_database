import os
import cv2

# Paths to your datasets
dataset_dirs = ['train', 'validation']

for dataset in dataset_dirs:
    image_dir = os.path.join(dataset, 'images')
    label_dir = os.path.join(dataset, 'labels')

    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        # Get image file name
        img_name = os.path.splitext(label_file)[0] + '.png'
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, label_file)

        # Load image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping missing image: {img_path}")
            continue

        h, w = img.shape[:2]

        # Read and normalize label
        new_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Skip malformed lines
                cls, x, y, bw, bh = parts
                x = float(x) / w
                y = float(y) / h
                bw = float(bw) / w
                bh = float(bh) / h
                new_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

        # Overwrite file with normalized values
        with open(label_path, 'w') as f:
            f.write('\n'.join(new_lines))
