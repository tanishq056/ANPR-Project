# Automatic Number Plate Recognition (ANPR) System

Host Link :- https://huggingface.co/spaces/tanishq04/ANPR ( Live Video stream may not visible on hosted server due to platform issue and it will take some time to load Output due to Low GPU )

OUTPUT VIDEO IF THE HOST LINK IS NOT WORKING :- https://drive.google.com/file/d/1a9yNxkHZCoU8TR_dTwxKKWRJ5L6HGSDn/view?usp=sharing

## Overview
This project is an Automatic Number Plate Recognition (ANPR) system using YOLO and Tesseract OCR. It processes a video file to detect and recognize license plates, then saves the detected plates to a CSV file and provides a live streaming interface via Flask.

## Features
- Uses YOLO for license plate detection.
- Uses Tesseract OCR for text extraction.
- Streams processed video with bounding boxes via Flask.
- Saves detected plates with timestamps to a CSV file.
- Provides API endpoints to retrieve detected plates and CSV data.

## Requirements
### Software Dependencies:
- Python 3.x
- OpenCV
- Pytesseract
- Pandas
- Flask
- Ultralytics YOLO
- Regex

### Install Dependencies:
```bash
pip install opencv-python pytesseract pandas flask ultralytics
```

### Setup Tesseract OCR:
Download and install Tesseract OCR from:
[Tesseract Download](https://github.com/UB-Mannheim/tesseract/wiki)

Set the Tesseract path in the script:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Project Structure
```
ANPR-Project/
│-- static/
│   ├── output_video.mp4  # Processed video output
│   ├── detected_plates.csv  # CSV storing detected plates
│-- templates/
│   ├── index.html  # Web interface
│-- main.py  # Main Flask app
```

## How to Run the Project
1. **Ensure YOLO Model is Available:**
   - Place the trained YOLO model weights in the specified path:
     ```python
     model = YOLO(r"D:\ANPR Project\ANPR\runs\detect\train9\weights\best.pt")
     ```
2. **Run the Flask App:**
   ```bash
   python main.py
   ```
3. **Open Web Interface:**
   - Navigate to `http://127.0.0.1:5000/` in your browser.
4. **Access API Endpoints:**
   - Live Video Stream: `http://127.0.0.1:5000/video_feed`
   - Get Detected Plates: `http://127.0.0.1:5000/get_detected_plates`
   - Get CSV Data: `http://127.0.0.1:5000/get_csv_data`

## Notes
- The processed video and detected plates CSV file will be saved in the `static/` directory.
- Make sure Tesseract OCR is properly installed and configured.
- Adjust video path and model weights path as needed.

