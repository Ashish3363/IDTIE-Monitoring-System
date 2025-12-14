# IDTieMonitor

A real-time facial recognition and uniform compliance monitoring system that detects students without proper ID cards or ties using camera feeds.

## Overview

IDTieMonitor combines facial recognition with object detection to monitor student compliance with uniform policies. The system uses YOLOv8 for face detection and object detection, ArcFace for facial recognition, and FAISS for efficient similarity searches.

**Frontend Repository:**  
https://github.com/09samuel/idtiemonitor-frontend

## Tech Stack

### Backend
- **FastAPI** - RESTful API server
- **Python 3.x** - Core backend language
- **MongoDB** - Student data storage
- **Redis** - Real-time index reloading
- **FAISS** - Vector similarity search for facial embeddings

### Frontend
- **Angular** - Single-page application framework
- **TypeScript** - Type-safe frontend development

### Machine Learning
- **YOLOv8** - Face detection and ID/tie object detection
- **ArcFace (DeepFace)** - Facial recognition and embeddings
- **OpenCV** - Image processing and camera feed handling
- **ByteTrack** - Multi-object tracking

## Features

- Real-time face detection and recognition from camera feeds
- ID card and tie presence detection
- Violation logging with screenshots
- Temporal smoothing with voting system for reliable recognition
- Low-light enhancement using CLAHE
- Hot-reload capability for updating face database
- Duplicate violation prevention



### Recognition Pipeline
1. Camera feed captures frames
2. CLAHE enhancement for low-light conditions
3. YOLOv8 detects faces with ByteTrack tracking
4. YOLOv8 detects ID cards and ties
5. Face images queued for recognition
6. ArcFace generates embeddings
7. FAISS performs similarity search
8. Voting system confirms identity
9. Violations logged to MongoDB with screenshots

## Installation

### Prerequisites
```bash
Python 3.8+
MongoDB
Redis
Node.js 16+
Angular CLI
```

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/09samuel/idtiemonitor.git
cd idtiemonitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download YOLO models
# Place yolov8s-face-lindevs.pt in project root
# Train custom ID/tie detector
```



## Usage

### Running the System

```bash

# Terminal 1: Start FastAPI backend
uvicorn backend:app --reload                                                                                                                                         

# Terminal 2: Start recognition engine
python python face_detector.py 
```

### Access Points
- **Frontend**: http://localhost:4200
- **API Docs**: http://localhost:8000/docs
- **Camera Feed**: OpenCV window (when recognition.py is running)



### Monitoring Violations

The system displays real-time feedback:
- **Green Box**: Recognized student, compliant
- **Red Box**: Violation detected (missing ID/tie)
- **Yellow Box**: Scanning/recognizing
- **Blue Box**: Queued for recognition

Violations are automatically:
- Logged to MongoDB
- Saved as screenshots in `violations/` folder
- Visible in frontend dashboard




## Performance Optimizations

- **Frame Skipping**: Process every N frames to reduce CPU load
- **Queue-based Recognition**: Asynchronous face recognition
- **FAISS Indexing**: O(log n) similarity search
- **ByteTrack**: Consistent face tracking across frames
- **Temporal Voting**: Multiple confirmations reduce false positives
- **Counter-based Logging**: Prevents duplicate violations
- **GPU Acceleration**: Use CUDA for YOLOv8 inference (if available)




## Troubleshooting

### Low Recognition Accuracy
- Adjust `EMBEDDING_THRESHOLD` (lower = stricter)
- Increase `CONFIRMATION_THRESHOLD` (more votes required)
- Add more diverse training photos per student
- Ensure good lighting conditions

### False Violation Alerts
- Increase `VIOLATION_LOG_THRESHOLD`
- Adjust `ID_TIE_CONF_THRESHOLD`
- Retrain ID/tie detector with more data
- Check camera angle and resolution

### Performance Issues
- Increase `PROCESS_EVERY_N_FRAMES`
- Reduce camera resolution
- Use GPU acceleration
- Optimize database queries


## Acknowledgments

- YOLOv8 by Ultralytics
- DeepFace library
- FAISS by Facebook Research
- ByteTrack for object tracking