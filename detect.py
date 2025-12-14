
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('images_dataset/runs/detect/train5/weights/best.pt')

# Open the laptop camera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Optional: set camera resolution
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO inference on the frame
    results = model(frame, conf=0.5)

    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Live", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
