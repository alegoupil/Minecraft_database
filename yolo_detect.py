import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
import mss

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source: file, folder, video, "usb0", "picamera0", or "screen"', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold (example: "0.4")', default=0.5)
parser.add_argument('--resolution', help='Resolution WxH for display (example: "1280x720")', default=None)
parser.add_argument('--record', help='Record results as demo1.avi. Requires --resolution.', action='store_true')
args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check model
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(0)

# Load model
model = YOLO(model_path, task='detect')
labels = model.names

# Detect input type
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file type: {ext}')
        sys.exit(0)
elif img_source == 'screen':
    source_type = 'screen'
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Invalid input source: {img_source}')
    sys.exit(0)

# Resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Recording
if record:
    if source_type not in ['video', 'usb', 'screen']:
        print('Recording only works for video, usb, or screen.')
        sys.exit(0)
    if not user_res:
        print('Specify resolution when recording.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# Input setup
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type == 'video' or source_type == 'usb':
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Inference loop
while True:
    t_start = time.perf_counter()

    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images processed.')
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Video or camera ended/disconnected.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Picamera not working.')
            break

    elif source_type == 'screen':
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            object_count += 1

    if source_type in ['video', 'usb', 'picamera', 'screen']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results', frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(1 if source_type in ['video', 'usb', 'picamera', 'screen'] else 0)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()