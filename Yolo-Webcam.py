from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialize the webcam (VideoCapture(0) opens the default camera)
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)  # Set the width of the webcam window
cap.set(4, 720)  # Set the height of the webcam window

# Uncomment the following line if you want to use a video file instead of a webcam
# cap = cv2.VideoCapture("../Videos/motorbikes-1.mp4")  # For Videos

# Load the YOLO model with the specified weights (yolov8n.pt in this case)
model = YOLO("../Yolo-Weights/yolov8n.pt")

# List of class names that the YOLO model can detect (COCO dataset classes)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Infinite loop to continuously read frames from the webcam/video
while True:
    success, img = cap.read()  # Capture the frame from the webcam
    results = model(img, stream=True)  # Pass the frame through the YOLO model and stream results

    # Process each detection result
    for r in results:
        boxes = r.boxes  # Get the bounding boxes for each detected object
        for box in boxes:
            # Get the coordinates of the bounding box (top-left and bottom-right corners)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers

            # Calculate the width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Draw a rectangle around the detected object with corner styling using cvzone
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Get the confidence score for the detected object
            conf = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to 2 decimal places

            # Get the class ID of the detected object
            cls = int(box.cls[0])

            # Display the class name and confidence score on the image
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Display the image with detections in a window
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # Wait for a key press to move to the next frame (1 ms delay)
