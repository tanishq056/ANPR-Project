import cv2
import pytesseract
import pandas as pd
import re
import os
from ultralytics import YOLO
from datetime import datetime

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the YOLO model
model = YOLO(r"D:\ANPR Project\ANPR\runs\detect\train9\weights\best.pt")

# Open the video file
cap = cv2.VideoCapture(r"D:\ANPR Project\videoplayback.mp4")

# Get video properties
frame_width = int(cap.get(3))  # Video width
frame_height = int(cap.get(4))  # Video height
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Define the output video writer
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# List to store detected plates
data = []

# Function to save license plates to a CSV file
def save_to_csv(number):
    global data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.append([number, timestamp])

    df = pd.DataFrame(data, columns=["License Plate", "Timestamp"])
    
    file_path = os.path.abspath("detected_plates.csv")  # Save in script directory
    try:
        df.to_csv(file_path, index=False)
        print(f"✅ CSV file saved successfully at: {file_path}")
    except Exception as e:
        print(f"❌ Error saving CSV file: {e}")  

# Define fast-forward speed (skip every 'n' frames)
fast_forward_speed = 5  # Adjust this to control fast-forward speed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection using YOLO
    results = model.predict(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]

            # Convert to grayscale
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            # Extract text using Tesseract
            text = pytesseract.image_to_string(gray).strip()

            # Filter only capital letters and numbers
            filtered_text = re.sub(r'[^A-Z0-9]', '', text)

            # If a valid plate is detected, save it
            if filtered_text:
                save_to_csv(filtered_text)

            # Draw bounding box and put text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, filtered_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the output frame
    cv2.imshow("License Plate Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Press 'q' to exit
        break
    elif key == ord(' '):  # Press 'space' to fast forward
        for _ in range(fast_forward_speed):
            cap.read()  # Skip frames

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Processed video saved at: {output_video_path}")
