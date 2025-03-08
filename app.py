import cv2
import pytesseract
import pandas as pd
import re
import os
from ultralytics import YOLO
from datetime import datetime
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO(r"D:\ANPR Project\ANPR\runs\detect\train9\weights\best.pt")

# Open video file
video_path = r"D:\ANPR Project\videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))  # Video width
frame_height = int(cap.get(4))  # Video height
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Define video writer
output_video_path = "static/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# List to store detected plates
detected_plates = []

# Function to save detected plates to CSV
def save_to_csv():
    file_path = os.path.abspath("static/detected_plates.csv")  # Save in static folder
    df = pd.DataFrame(detected_plates, columns=["License Plate", "Timestamp"])
    
    try:
        df.to_csv(file_path, index=False)
        print(f"✅ CSV saved successfully: {file_path}")
    except Exception as e:
        print(f"❌ Error saving CSV file: {e}")  

# Video processing function
def process_video():
    global detected_plates
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

                # Filter valid number plates
                filtered_text = re.sub(r'[^A-Z0-9]', '', text)

                # Save detected plates
                if filtered_text and filtered_text not in [p[0] for p in detected_plates]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detected_plates.append([filtered_text, timestamp])

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, filtered_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write processed frame to video
        out.write(frame)

        # Encode frame for web streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame to web page
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()
    save_to_csv()  # Save CSV when video processing is done

# Flask Route: Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Flask Route: Stream Video
@app.route('/video_feed')
def video_feed():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask Route: Get Detected Plates
@app.route('/get_detected_plates')
def get_detected_plates():
    return jsonify(detected_plates)

# Flask Route: Get CSV Data
@app.route('/get_csv_data')
def get_csv_data():
    csv_path = "static/detected_plates.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.to_html(classes="table table-bordered")
    return "<h3>No Data Available</h3>"

if __name__ == '__main__':
    app.run(debug=True)
