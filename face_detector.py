# import cv2
# import threading
# from queue import Queue
# from ultralytics import YOLO
# from deepface import DeepFace
# import numpy as np

# # -------------------------------
# # Configuration
# # -------------------------------
# YOLO_MODEL_PATH = "yolov8s-face-lindevs.pt"
# VIDEO_SOURCE = 0
# FACE_DB_PATH = "faces_db/"
# EMBEDDING_THRESHOLD = 0.4  # Threshold for considering same person

# # Cache to store embeddings and names
# embedding_cache = []  # list of tuples (embedding, name)

# # Queue for recognition
# recognition_queue = Queue()

# # -------------------------------
# # Thread function for DeepFace recognition
# # -------------------------------
# def recognize_face_thread():
#     while True:
#         item = recognition_queue.get()
#         if item is None:
#             break
#         face_img = item
#         try:
#             result = DeepFace.find(
#                 face_img,
#                 db_path=FACE_DB_PATH,
#                 model_name="SFace",
#                 detector_backend="skip",
#                 enforce_detection=False
#             )
#             if not result.empty:
#                 person_name = result.iloc[0]["identity"].split("/")[-1].replace(".jpg", "")
#             else:
#                 person_name = "Unknown"
#         except:
#             person_name = "Unknown"

#         # Compute embedding for caching
#         try:
#             embedding = DeepFace.represent(face_img, model_name="SFace", enforce_detection=False)[0]["embedding"]
#             embedding_cache.append((embedding, person_name))
#         except:
#             pass

#         recognition_queue.task_done()

# # Start recognition thread
# threading.Thread(target=recognize_face_thread, daemon=True).start()

# # -------------------------------
# # Open video source
# # -------------------------------
# cap = cv2.VideoCapture(VIDEO_SOURCE)
# face_detector = YOLO(YOLO_MODEL_PATH)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_resized = cv2.resize(frame, (640, 480))

#     # Detect faces
#     results = face_detector(frame_resized)

#     for box in results[0].boxes.xyxy:
#         x1, y1, x2, y2 = map(int, box)
#         face_img = frame_resized[y1:y2, x1:x2]

#         # Compute embedding for current face
#         try:
#             embedding = DeepFace.represent(face_img, model_name="SFace", enforce_detection=False)[0]["embedding"]
#         except:
#             embedding = None

#         name = "Recognizing..."
#         if embedding is not None:
#             # Compare with cached embeddings
#             for emb, cached_name in embedding_cache:
#                 dist = np.linalg.norm(np.array(embedding) - np.array(emb))
#                 if dist < EMBEDDING_THRESHOLD:
#                     name = cached_name
#                     break
#             else:
#                 # If not found, send to recognition thread
#                 recognition_queue.put(face_img)

#         # Draw bounding box and name
#         cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame_resized, name, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     cv2.imshow("Face Recognition", frame_resized)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
# recognition_queue.put(None)





























# import cv2
# from ultralytics import YOLO

# # Load YOLO face detector
# face_detector = YOLO("yolov8s-face-lindevs.pt")

# # Open webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect faces
#     results = face_detector(frame)

#     for box in results[0].boxes.xyxy:
#         x1, y1, x2, y2 = map(int, box)
#         # Draw bounding box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     cv2.imshow("Webcam Feed", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()







# import cv2
# from ultralytics import YOLO
# from deepface import DeepFace
# import os
# import time

# # -------------------------------
# # Configuration
# # -------------------------------
# FACE_DB_PATH = "faces_db/"
# EMBEDDING_THRESHOLD = 12.0  # Distance threshold for recognition

# # -------------------------------
# # Load YOLO face detector
# # -------------------------------
# face_detector = YOLO("yolov8s-face-lindevs.pt")

# # Open webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = face_detector(frame)

#     for box in results[0].boxes.xyxy:
#         x1, y1, x2, y2 = map(int, box)
#         face_img = frame[y1:y2, x1:x2]

#         # Perform recognition
#         person_name = "Unknown"
#         try:
#             result = DeepFace.find(
#                 face_img,
#                 db_path=FACE_DB_PATH,
#                 model_name="SFace",
#                 detector_backend="skip",
#                 enforce_detection=False,
#                 distance_metric="euclidean",
#                 silent=True
#             )
            
#             if isinstance(result, list) and len(result) > 0:
#                 df = result[0]
#                 if not df.empty:
#                     best_match = df.loc[df['distance'].idxmin()]
#                     distance = best_match['distance']
#                     if distance < EMBEDDING_THRESHOLD:
#                         identity_path = best_match["identity"]
#                         person_name = identity_path.split(os.sep)[-2]
#         except Exception as e:
#             person_name = "Error"

#         # Draw bounding box and name
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, person_name, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow("Webcam Feed", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()














































































#WORKING COPY

# import cv2
# from ultralytics import YOLO
# from deepface import DeepFace
# import numpy as np
# import threading
# from queue import Queue
# import os
# import time

# # -------------------------------
# # Configuration
# # -------------------------------
# FACE_DB_PATH = "faces_db/"
# EMBEDDING_THRESHOLD = 13.0  # Slightly higher for distant faces
# MIN_FACE_SIZE = 50          # Minimum face size to process
# MAX_QUEUE_SIZE = 8
# RECOGNITION_TIMEOUT = 8
# PROCESS_EVERY_N_FRAMES = 3  # Reduce load by processing every N frames

# # -------------------------------
# # Global variables
# # -------------------------------
# recognition_results = {}   # Stores recognition info per face ID
# processing_faces = {}      # Faces currently being processed
# recognition_queue = Queue()

# face_tracker = {}          # Tracks face positions and IDs
# next_face_id = 1
# frame_count = 0

# # -------------------------------
# # Helper functions
# # -------------------------------
# def get_face_center(x1, y1, x2, y2):
#     return ((x1 + x2) // 2, (y1 + y2) // 2)

# def find_matching_face_id(current_center, current_size):
#     """Assign a consistent ID based on position"""
#     global face_tracker, next_face_id
#     best_match = None
#     min_distance = 100

#     for fid, (center, size, last_seen) in face_tracker.items():
#         if time.time() - last_seen > 2:  # Skip old faces
#             continue
#         dist = np.linalg.norm(np.array(center) - np.array(current_center))
#         size_diff = abs(size - current_size)
#         if dist < min_distance and size_diff < 50:
#             min_distance = dist
#             best_match = fid

#     if best_match:
#         face_tracker[best_match] = (current_center, current_size, time.time())
#         return best_match
#     else:
#         new_id = f"person_{next_face_id}"
#         next_face_id += 1
#         face_tracker[new_id] = (current_center, current_size, time.time())
#         return new_id

# # -------------------------------
# # Recognition thread
# # -------------------------------
# def recognize_face_thread():
#     while True:
#         item = recognition_queue.get()
#         if item is None:
#             break
#         face_id, face_img, start_time = item
#         try:
#             result = DeepFace.find(
#                 face_img,
#                 db_path=FACE_DB_PATH,
#                 model_name="Facenet",
#                 detector_backend="skip",
#                 enforce_detection=False,
#                 distance_metric="euclidean",
#                 silent=True
#             )

#             person_name = "Unknown"
#             distance = float('inf')

#             if isinstance(result, list) and len(result) > 0:
#                 df = result[0]
#                 if not df.empty:
#                     best_match = df.loc[df['distance'].idxmin()]
#                     distance = best_match['distance']
#                     if distance < EMBEDDING_THRESHOLD:
#                         identity_path = best_match["identity"]
#                         person_name = identity_path.split(os.sep)[-2]

#             recognition_results[face_id] = {
#                 'name': person_name,
#                 'distance': distance,
#                 'timestamp': time.time()
#             }
#         except Exception as e:
#             recognition_results[face_id] = {
#                 'name': "Error",
#                 'distance': float('inf'),
#                 'timestamp': time.time()
#             }
#         finally:
#             if face_id in processing_faces:
#                 del processing_faces[face_id]
#         recognition_queue.task_done()

# # -------------------------------
# # Queue face for recognition
# # -------------------------------
# def queue_face(face_id, face_img):
#     if face_id not in processing_faces and recognition_queue.qsize() < MAX_QUEUE_SIZE:
#         processing_faces[face_id] = time.time()
#         recognition_queue.put((face_id, face_img.copy(), time.time()))

# # -------------------------------
# # Cleanup old results
# # -------------------------------
# def cleanup_old_results():
#     current_time = time.time()
#     # Remove old recognition results
#     old_results = [fid for fid, res in recognition_results.items() if current_time - res['timestamp'] > 10]
#     for fid in old_results:
#         del recognition_results[fid]
#     # Remove old tracked faces
#     old_faces = [fid for fid, (_, _, last_seen) in face_tracker.items() if current_time - last_seen > 5]
#     for fid in old_faces:
#         del face_tracker[fid]

# # -------------------------------
# # Main execution
# # -------------------------------
# if __name__ == "__main__":
#     # Start recognition thread
#     threading.Thread(target=recognize_face_thread, daemon=True).start()

#     # Load YOLO
#     face_detector = YOLO("yolov8s-face-lindevs.pt")
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         frame_resized = cv2.resize(frame, (640, 480))
#         should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

#         results = face_detector(frame_resized)

#         for box in results[0].boxes.xyxy:
#             x1, y1, x2, y2 = map(int, box)
#             w, h = x2 - x1, y2 - y1
#             # if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
#             #     continue
            

#             face_center = get_face_center(x1, y1, x2, y2)
#             face_size = max(w, h)
#             face_id = find_matching_face_id(face_center, face_size)

#             face_img = frame_resized[y1:y2, x1:x2]
#             name = "Scanning..."

#             if face_id in recognition_results:
#                 res = recognition_results[face_id]
#                 name = res['name']

#             elif should_process:
#                 queue_face(face_id, face_img)

#             # Draw bounding box and name
#             cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame_resized, name, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             cv2.putText(frame_resized, face_id, (x1, y1 + 15),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

#         # Cleanup old entries
#         cleanup_old_results()

#         cv2.imshow("CCTV Recognition", frame_resized)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     recognition_queue.put(None)































































# import cv2
# import threading
# from queue import Queue
# from ultralytics import YOLO
# from deepface import DeepFace
# import numpy as np
# import os
# import time

# # -------------------------------
# # Configuration - OPTIMIZED
# # -------------------------------
# YOLO_MODEL_PATH = "yolov8s-face-lindevs.pt"
# VIDEO_SOURCE = 0
# FACE_DB_PATH = "faces_db/"

# # ADJUSTED THRESHOLDS based on your output
# EMBEDDING_THRESHOLD = 12.0  # Increased from 0.6 (your face had 9.237)
# MIN_FACE_SIZE = 50
# RECOGNITION_TIMEOUT = 8
# MAX_QUEUE_SIZE = 8  # Increased queue size
# PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame to reduce load

# # Global variables
# recognition_results = {}
# processing_faces = {}
# face_counter = 0
# recognition_queue = Queue()
# last_processed_frame = 0

# # Face tracking for consistent IDs
# face_tracker = {}  # Store face positions for tracking
# next_face_id = 1

# # -------------------------------
# # Face tracking helper
# # -------------------------------
# def get_face_center(x1, y1, x2, y2):
#     return ((x1 + x2) // 2, (y1 + y2) // 2)

# def find_matching_face_id(current_center, current_size):
#     """Find existing face ID based on position similarity"""
#     global face_tracker, next_face_id
    
#     best_match = None
#     min_distance = 100  # pixels
    
#     for face_id, (center, size, last_seen) in face_tracker.items():
#         # Skip faces not seen recently
#         if time.time() - last_seen > 2:
#             continue
            
#         # Calculate distance between centers
#         dist = np.sqrt((current_center[0] - center[0])*2 + (current_center[1] - center[1])*2)
#         size_diff = abs(current_size - size)
        
#         if dist < min_distance and size_diff < 50:
#             min_distance = dist
#             best_match = face_id
    
#     if best_match:
#         # Update tracking info
#         face_tracker[best_match] = (current_center, current_size, time.time())
#         return best_match
#     else:
#         # Create new face ID
#         new_id = f"person_{next_face_id}"
#         next_face_id += 1
#         face_tracker[new_id] = (current_center, current_size, time.time())
#         return new_id

# # -------------------------------
# # Optimized recognition thread
# # -------------------------------
# def recognize_face_thread():
#     while True:
#         item = recognition_queue.get()
#         if item is None:
#             break
        
#         face_id, face_img, start_time = item
        
#         try:
#             print(f"🔍 Processing {face_id}...")
            
#             # Search database with optimized settings
#             result = DeepFace.find(
#                 face_img,
#                 db_path=FACE_DB_PATH,
#                 model_name="SFace",
#                 detector_backend="skip",
#                 enforce_detection=False,
#                 distance_metric="euclidean",
#                 silent=True  # Reduce console spam
#             )
            
#             person_name = "Unknown"
#             distance = float('inf')
            
#             if isinstance(result, list) and len(result) > 0:
#                 df = result[0]
#                 if not df.empty:
#                     best_match = df.loc[df['distance'].idxmin()]
#                     distance = best_match['distance']
                    
#                     print(f"📊 Best match distance: {distance:.3f} (threshold: {EMBEDDING_THRESHOLD})")
                    
#                     if distance < EMBEDDING_THRESHOLD:
#                         identity_path = best_match["identity"]
#                         person_name = identity_path.split(os.sep)[-2]
#                         print(f"✅ Recognized: {person_name}")
#                     else:
#                         print(f"❌ Distance {distance:.3f} > threshold {EMBEDDING_THRESHOLD}")
            
#             total_time = time.time() - start_time
            
#             recognition_results[face_id] = {
#                 'name': person_name,
#                 'distance': distance,
#                 'processing_time': total_time,
#                 'timestamp': time.time()
#             }
            
#         except Exception as e:
#             print(f"❌ Recognition error for {face_id}: {e}")
#             recognition_results[face_id] = {
#                 'name': "Error",
#                 'distance': float('inf'),
#                 'processing_time': time.time() - start_time,
#                 'timestamp': time.time()
#             }
        
#         finally:
#             if face_id in processing_faces:
#                 del processing_faces[face_id]
        
#         recognition_queue.task_done()

# # -------------------------------
# # Cleanup old results
# # -------------------------------
# def cleanup_old_results():
#     """Remove old recognition results to prevent memory buildup"""
#     current_time = time.time()
#     old_results = [face_id for face_id, result in recognition_results.items() 
#                    if current_time - result['timestamp'] > 10]
    
#     for face_id in old_results:
#         del recognition_results[face_id]
    
#     # Cleanup face tracker
#     old_faces = [face_id for face_id, (_, _, last_seen) in face_tracker.items()
#                  if current_time - last_seen > 5]
    
#     for face_id in old_faces:
#         del face_tracker[face_id]

# # -------------------------------
# # Main execution
# # -------------------------------
# if _name_ == "_main_":
#     print("=== Optimized Face Recognition ===")
#     print(f"Using threshold: {EMBEDDING_THRESHOLD}")
    
#     # Start recognition thread
#     recognition_thread = threading.Thread(target=recognize_face_thread, daemon=True)
#     recognition_thread.start()
    
#     # Initialize video
#     cap = cv2.VideoCapture(VIDEO_SOURCE)
#     if not cap.isOpened():
#         print("❌ Could not open video source")
#         exit()
    
#     face_detector = YOLO(YOLO_MODEL_PATH)
    
#     print("🎥 Starting optimized recognition...")
#     print("Press 'q' to quit, 'c' to clear results")
    
#     frame_count = 0
#     last_cleanup = time.time()
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
#         frame_resized = cv2.resize(frame, (640, 480))
        
#         # Process every N frames to reduce load
#         should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)
        
#         # Detect faces
#         results = face_detector(frame_resized)
        
#         # Cleanup every 30 seconds
#         if time.time() - last_cleanup > 30:
#             cleanup_old_results()
#             last_cleanup = time.time()
        
#         # Clean up timeouts
#         current_time = time.time()
#         timeout_faces = [fid for fid, start_time in processing_faces.items() 
#                         if current_time - start_time > RECOGNITION_TIMEOUT]
#         for fid in timeout_faces:
#             del processing_faces[fid]
#             recognition_results[fid] = {
#                 'name': "Timeout", 
#                 'distance': float('inf'), 
#                 'processing_time': RECOGNITION_TIMEOUT,
#                 'timestamp': current_time
#             }
        
#         if len(results) > 0 and results[0].boxes is not None:
#             for i, box in enumerate(results[0].boxes.xyxy):
#                 x1, y1, x2, y2 = map(int, box)
                
#                 # Check minimum face size
#                 face_width, face_height = x2 - x1, y2 - y1
#                 if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
#                     continue
                
#                 # Get consistent face ID based on position
#                 face_center = get_face_center(x1, y1, x2, y2)
#                 face_size = max(face_width, face_height)
#                 consistent_face_id = find_matching_face_id(face_center, face_size)
                
#                 face_img = frame_resized[y1:y2, x1:x2]
                
#                 name = "Scanning..."
#                 color = (128, 128, 128)  # Gray
#                 status_text = ""
                
#                 # Check if we have a result for this consistent face ID
#                 if consistent_face_id in recognition_results:
#                     result = recognition_results[consistent_face_id]
#                     name = result['name']
#                     distance = result['distance']
#                     processing_time = result['processing_time']
                    
#                     if distance < EMBEDDING_THRESHOLD and name != "Unknown":
#                         status_text = f" (d:{distance:.1f})"
#                         color = (0, 255, 0)  # Green
#                     elif name in ["Error", "Timeout"]:
#                         color = (0, 0, 255)  # Red
#                     else:
#                         status_text = f" (d:{distance:.1f})"
#                         color = (0, 165, 255)  # Orange
                
#                 # Check if currently processing
#                 elif consistent_face_id in processing_faces:
#                     elapsed = current_time - processing_faces[consistent_face_id]
#                     name = f"Recognizing... {elapsed:.1f}s"
#                     color = (255, 255, 0)  # Cyan
                
#                 # Start new recognition if conditions are met
#                 elif (should_process and 
#                       recognition_queue.qsize() < MAX_QUEUE_SIZE and
#                       face_width > 80 and face_height > 80):  # Only process larger faces
                    
#                     processing_faces[consistent_face_id] = current_time
#                     recognition_queue.put((consistent_face_id, face_img.copy(), current_time))
#                     name = "Queued..."
#                     color = (255, 0, 0)  # Blue
                
#                 elif recognition_queue.qsize() >= MAX_QUEUE_SIZE:
#                     name = f"Queue full ({recognition_queue.qsize()})"
#                     color = (128, 128, 128)  # Gray
                
#                 # Draw bounding box and info
#                 cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                
#                 # Face info
#                 cv2.putText(frame_resized, name + status_text, (x1, y1 - 10),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
#                 # Face ID for tracking
#                 cv2.putText(frame_resized, consistent_face_id, (x1, y1 + 15),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         # Status overlay
#         status_lines = [
#             f"Queue: {recognition_queue.qsize()}/{MAX_QUEUE_SIZE}",
#             f"Processing: {len(processing_faces)}",
#             f"Results: {len(recognition_results)}",
#             f"Threshold: {EMBEDDING_THRESHOLD}"
#         ]
        
#         for i, line in enumerate(status_lines):
#             cv2.putText(frame_resized, line, (10, 20 + i*20),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         cv2.imshow("Optimized Face Recognition", frame_resized)
        
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         elif key == ord("c"):
#             # Clear all results
#             recognition_results.clear()
#             processing_faces.clear()
#             face_tracker.clear()
#             print("🔄 Cleared all results")
    
#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
#     recognition_queue.put(None)
#     print("👋 Application closed")












# import cv2
# import threading
# from queue import Queue
# from ultralytics import YOLO
# from deepface import DeepFace
# import numpy as np
# import os
# import time

# # -------------------------------
# # Configuration
# # -------------------------------
# YOLO_MODEL_PATH = "yolov8s-face-lindevs.pt"
# VIDEO_SOURCE = 0
# FACE_DB_PATH = "faces_db/"
# EMBEDDING_THRESHOLD = 12.0
# MIN_FACE_SIZE = 50
# RECOGNITION_TIMEOUT = 8
# MAX_QUEUE_SIZE = 6
# PROCESS_EVERY_N_FRAMES = 2

# # Global variables
# recognition_results = {}
# processing_faces = {}
# recognition_queue = Queue()
# face_tracker = {}
# next_face_id = 1

# # -------------------------------
# # Face tracking helper
# # -------------------------------
# def get_face_center(x1, y1, x2, y2):
#     return ((x1 + x2) // 2, (y1 + y2) // 2)

# def find_matching_face_id(center, size):
#     global face_tracker, next_face_id
#     best_match = None
#     min_distance = 100
#     for fid, (c, s, last_seen) in face_tracker.items():
#         if time.time() - last_seen > 2: 
#             continue
#         dist = np.linalg.norm(np.array(center) - np.array(c))
#         if dist < min_distance and abs(size - s) < 50:
#             min_distance = dist
#             best_match = fid
#     if best_match:
#         face_tracker[best_match] = (center, size, time.time())
#         return best_match
#     else:
#         fid = f"person_{next_face_id}"
#         next_face_id += 1
#         face_tracker[fid] = (center, size, time.time())
#         return fid

# # -------------------------------
# # Recognition thread
# # -------------------------------
# def recognize_face_thread():
#     while True:
#         item = recognition_queue.get()
#         if item is None:
#             break
#         face_id, face_img, start_time = item
#         try:
#             result = DeepFace.find(face_img, db_path=FACE_DB_PATH, model_name="SFace",
#                                    detector_backend="skip", enforce_detection=False,
#                                    distance_metric="euclidean", silent=True)
#             name = "Unknown"
#             distance = float('inf')
#             if isinstance(result, list) and len(result) > 0:
#                 df = result[0]
#                 if not df.empty:
#                     best_match = df.loc[df['distance'].idxmin()]
#                     distance = best_match['distance']
#                     if distance < EMBEDDING_THRESHOLD:
#                         name = best_match["identity"].split(os.sep)[-2]
#             recognition_results[face_id] = {'name': name, 'distance': distance, 'timestamp': time.time()}
#         except Exception as e:
#             recognition_results[face_id] = {'name': "Error", 'distance': float('inf'), 'timestamp': time.time()}
#         finally:
#             if face_id in processing_faces:
#                 del processing_faces[face_id]
#         recognition_queue.task_done()

# # -------------------------------
# # Cleanup old results
# # -------------------------------
# def cleanup_old_results():
#     now = time.time()
#     for fid in list(recognition_results):
#         if now - recognition_results[fid]['timestamp'] > 10:
#             del recognition_results[fid]
#     for fid in list(face_tracker):
#         if now - face_tracker[fid][2] > 5:
#             del face_tracker[fid]

# # -------------------------------
# # Main execution
# # -------------------------------
# if _name_ == "_main_":
#     recognition_thread = threading.Thread(target=recognize_face_thread, daemon=True)
#     recognition_thread.start()

#     cap = cv2.VideoCapture(VIDEO_SOURCE)
#     if not cap.isOpened():
#         print("❌ Could not open video source")
#         exit()

#     face_detector = YOLO(YOLO_MODEL_PATH)
#     frame_count = 0
#     last_cleanup = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         frame_resized = cv2.resize(frame, (640, 480))
#         should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

#         results = face_detector(frame_resized)

#         if time.time() - last_cleanup > 30:
#             cleanup_old_results()
#             last_cleanup = time.time()

#         current_time = time.time()
#         timeout_faces = [fid for fid, start in processing_faces.items() if current_time - start > RECOGNITION_TIMEOUT]
#         for fid in timeout_faces:
#             del processing_faces[fid]
#             recognition_results[fid] = {'name': "Timeout", 'distance': float('inf'), 'timestamp': current_time}

#         if len(results) > 0 and results[0].boxes is not None:
#             for box in results[0].boxes.xyxy:
#                 x1, y1, x2, y2 = map(int, box)
#                 w, h = x2 - x1, y2 - y1
#                 if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
#                     continue

#                 face_center = get_face_center(x1, y1, x2, y2)
#                 face_size = max(w, h)
#                 fid = find_matching_face_id(face_center, face_size)
#                 face_img = frame_resized[y1:y2, x1:x2]

#                 name = "Scanning..."
#                 color = (128, 128, 128)

#                 if fid in recognition_results:
#                     result = recognition_results[fid]
#                     name = result['name']
#                     if result['distance'] < EMBEDDING_THRESHOLD and name != "Unknown":
#                         color = (0, 255, 0)
#                     elif name in ["Error", "Timeout"]:
#                         color = (0, 0, 255)
#                     else:
#                         color = (0, 165, 255)
#                 elif fid in processing_faces:
#                     name = "Recognizing..."
#                     color = (255, 255, 0)
#                 elif should_process and recognition_queue.qsize() < MAX_QUEUE_SIZE:
#                     processing_faces[fid] = current_time
#                     recognition_queue.put((fid, face_img.copy(), current_time))
#                     name = "Queued..."
#                     color = (255, 0, 0)

#                 cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame_resized, name, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         status_lines = [
#             f"Queue: {recognition_queue.qsize()}/{MAX_QUEUE_SIZE}",
#             f"Processing: {len(processing_faces)}",
#             f"Results: {len(recognition_results)}"
#         ]
#         for i, line in enumerate(status_lines):
#             cv2.putText(frame_resized, line, (10, 20 + i*20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         cv2.imshow("Face Recognition (YOLO + SFace)", frame_resized)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         elif key == ord("c"):
#             recognition_results.clear()
#             processing_faces.clear()
#             face_tracker.clear()
#             print("🔄 Cleared all results")

#     cap.release()
#     cv2.destroyAllWindows()
#     recognition_queue.put(None)
#     print("👋 Application closed")











##WORKING BETTER WITH VOTING SYSTEM AND LOW LIGHT ENHANCEMENT
# import cv2
# from ultralytics import YOLO
# from deepface import DeepFace
# import numpy as np
# import threading
# from queue import Queue
# import os
# import time
# from collections import deque # NEW: Import deque

# # -------------------------------
# # Configuration
# # -------------------------------
# FACE_DB_PATH = "faces_db/"
# EMBEDDING_THRESHOLD = 0.68  # Using ArcFace + Cosine
# # NEW: Parameters for the Voting System
# REC_HISTORY_LIMIT = 10      # Store the last 10 recognition results for each person
# CONFIRMATION_THRESHOLD = 4  # A name must appear at least 4 times to be confirmed
# # NEW: YOLO confidence for face detection
# YOLO_CONF_THRESHOLD = 0.10  # Lower this to detect more, smaller faces
# # MIN_FACE_SIZE = 40          # Lowered this slightly
# # ... other configs
# MAX_QUEUE_SIZE = 8
# PROCESS_EVERY_N_FRAMES = 1  # Process more frequently

# CLAHE_CLIP_LIMIT = 2.0
# CLAHE_TILE_GRID_SIZE = (8, 8)

# # -------------------------------
# # Global variables
# # -------------------------------
# recognition_results = {}
# processing_faces = set()
# recognition_queue = Queue()

# # MODIFIED: face_tracker now stores a deque for recognition history
# face_tracker = {} # fid -> (center, size, last_seen, rec_history_deque)
# next_face_id = 1
# frame_count = 0

# # -------------------------------
# # Helper functions
# # -------------------------------
# def get_face_center(x1, y1, x2, y2):
#     return ((x1 + x2) // 2, (y1 + y2) // 2)

# # MODIFIED: Now initializes the deque for new faces
# def find_matching_face_id(current_center, current_size):
#     global face_tracker, next_face_id
#     best_match_id = None
#     min_distance = 100

#     for fid, (center, size, last_seen, _) in face_tracker.items():
#         if time.time() - last_seen > 2.0:
#             continue
#         dist = np.linalg.norm(np.array(center) - np.array(current_center))
#         if dist < min_distance:
#             min_distance = dist
#             best_match_id = fid

#     if best_match_id:
#         # Update everything but the history deque
#         _, size, _, rec_history = face_tracker[best_match_id]
#         face_tracker[best_match_id] = (current_center, current_size, time.time(), rec_history)
#         return best_match_id
#     else:
#         new_id = f"person_{next_face_id}"
#         next_face_id += 1
#         # When creating a new face, add an empty deque for its history
#         face_tracker[new_id] = (current_center, current_size, time.time(), deque(maxlen=REC_HISTORY_LIMIT))
#         return new_id
    

# def enhance_low_light(frame):
#     """Enhances a low-light frame using CLAHE on the L-channel of the LAB color space."""
#     # Convert the frame to the LAB color space
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
#     # Split the LAB image into L, A, and B channels
#     l_channel, a_channel, b_channel = cv2.split(lab)
    
#     # Create a CLAHE object
#     clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    
#     # Apply CLAHE to the L-channel
#     cl = clahe.apply(l_channel)
    
#     # Merge the CLAHE enhanced L-channel with the original A and B channels
#     merged = cv2.merge((cl, a_channel, b_channel))
    
#     # Convert the LAB image back to BGR color space
#     enhanced_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
#     return enhanced_frame

# # -------------------------------
# # Recognition thread (MODIFIED to update the history deque)
# # -------------------------------
# def recognize_face_thread():
#     while True:
#         item = recognition_queue.get()
#         if item is None: break
#         face_id, face_img = item
#         try:
#             result = DeepFace.find(
#                 face_img, db_path=FACE_DB_PATH, model_name="ArcFace",
#                 distance_metric="cosine", detector_backend="skip",
#                 enforce_detection=False, silent=True
#             )
#             person_name = "Unknown"
#             if isinstance(result, list) and len(result) > 0:
#                 df = result[0]
#                 if not df.empty:
#                     best_match = df.loc[df['distance'].idxmin()]
#                     if best_match['distance'] < EMBEDDING_THRESHOLD:
#                         identity_path = best_match["identity"]
#                         person_name = os.path.basename(os.path.dirname(identity_path))
            
#             # MODIFIED: Instead of a separate dict, update the deque in the tracker
#             if face_id in face_tracker:
#                 face_tracker[face_id][3].append(person_name)

#         except Exception as e:
#             print(f"Error in recognition for {face_id}: {e}")
#         finally:
#             if face_id in processing_faces:
#                 processing_faces.remove(face_id)
#             recognition_queue.task_done()

# # (queue_face_for_recognition and cleanup_old_data remain mostly the same,
# # but cleanup should now also handle the new face_tracker structure)

# def cleanup_old_data():
#     current_time = time.time()
#     old_faces = [fid for fid, (_, _, last_seen, _) in face_tracker.items() if current_time - last_seen > 5.0]
#     for fid in old_faces:
#         del face_tracker[fid]
#         if fid in processing_faces:
#             processing_faces.remove(fid)

# # ... (queue_face_for_recognition is the same as before) ...
# def queue_face_for_recognition(face_id, face_img):
#     if face_id in processing_faces: return
#     if recognition_queue.qsize() < MAX_QUEUE_SIZE:
#         processing_faces.add(face_id)
#         recognition_queue.put((face_id, face_img.copy()))

# # -------------------------------
# # Main execution
# # -------------------------------
# if __name__ == "__main__":
#     threading.Thread(target=recognize_face_thread, daemon=True).start()
#     face_detector = YOLO("yolov8s-face-lindevs.pt")
#     cap = cv2.VideoCapture(0)
#     # OPTIONAL: Try to set a higher resolution
#     # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     while True:
#         ret, frame = cap.read()
#         if not ret: break

#         enhanced_frame = enhance_low_light(frame.copy())

#         frame_count += 1
#         scale = 640 / enhanced_frame.shape[1]
#         dim = (640, int(enhanced_frame.shape[0] * scale))
#         frame_resized = cv2.resize(enhanced_frame, dim, interpolation=cv2.INTER_AREA)
        
#         should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

#         # MODIFIED: Added `conf` parameter to YOLO call
#         results = face_detector(frame_resized, verbose=False, conf=YOLO_CONF_THRESHOLD)

#         for box in results[0].boxes.xyxy:
#             x1, y1, x2, y2 = map(int, box)
#             w, h = x2 - x1, y2 - y1
            
#             # if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
#             #     continue

#             face_center = get_face_center(x1, y1, x2, y2)
#             face_id = find_matching_face_id(face_center, w)

#             if should_process:
#                 face_img = frame_resized[y1:y2, x1:x2]
#                 queue_face_for_recognition(face_id, face_img)

#             # --- NEW: Voting Logic for Display ---
#             name_to_display = "Scanning..."
#             color = (0, 0, 255) # Red for unknown/scanning
#             if face_id in face_tracker:
#                 rec_history = face_tracker[face_id][3]
#                 if rec_history:
#                     # Find the most common name in the history
#                     most_common_name = max(set(rec_history), key=rec_history.count)
#                     # Check if it meets the confirmation threshold
#                     if rec_history.count(most_common_name) >= CONFIRMATION_THRESHOLD:
#                         if most_common_name != "Unknown":
#                             name_to_display = most_common_name
#                             color = (0, 255, 0) # Green for confirmed
#                         else:
#                             name_to_display = "Unknown"

#             # Draw bounding box and name
#             cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame_resized, name_to_display, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#             cv2.putText(frame_resized, face_id, (x1, y2 + 15),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

#         if frame_count % 30 == 0:
#             cleanup_old_data()

#         cv2.imshow("CCTV Recognition", frame_resized)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     recognition_queue.put(None)



# # ABOVE ONE BUT WITH BYTETRACK
# import cv2
# from ultralytics import YOLO
# from deepface import DeepFace
# import numpy as np
# import threading
# from queue import Queue
# import os
# import time
# from collections import deque
# from pymongo import MongoClient
# import redis
# from threading import Lock
# from numpy.linalg import norm
# from recognition import load_index, recognize_face
# from backend import  Violation
# from recognition import add_violation


# # -------------------------------
# # MongoDB Setup
# # -------------------------------

# # MongoDB Config
# MONGO_URI = "mongodb://localhost:27017/"
# DB_NAME = "schoolDB"
# COLLECTION_NAME = "students"

# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# students_col = db[COLLECTION_NAME]

# def get_all_student_folders():
#     """Fetch all registered student photo folders"""
#     return [doc["photoFolder"] for doc in students_col.find()]

# # -------------------------------
# # Configuration
# # -------------------------------
# #FACE_DB_PATH = "faces_db/"
# EMBEDDING_THRESHOLD = 0.35  # ArcFace + Cosine
# REC_HISTORY_LIMIT = 10
# CONFIRMATION_THRESHOLD = 3
# YOLO_CONF_THRESHOLD = 0.10
# MAX_QUEUE_SIZE = 8
# PROCESS_EVERY_N_FRAMES = 1

# CLAHE_CLIP_LIMIT = 2.0
# CLAHE_TILE_GRID_SIZE = (8, 8)

# # -------------------------------
# # Global variables
# # -------------------------------
# processing_faces = set()
# recognition_queue = Queue()
# tracker_lock = Lock()


# # tracker: track_id -> {"last_seen": float, "rec_history": deque}
# face_tracker = {}
# frame_count = 0

# # -------------------------------
# # Helper functions
# # -------------------------------
# def enhance_low_light(frame):
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l_channel, a_channel, b_channel = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
#     cl = clahe.apply(l_channel)
#     merged = cv2.merge((cl, a_channel, b_channel))
#     return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# def queue_face_for_recognition(track_id, face_img):
#     if track_id in processing_faces:
#         return
#     if recognition_queue.qsize() < MAX_QUEUE_SIZE:
#         processing_faces.add(track_id)
#         recognition_queue.put((track_id, face_img.copy()))

# def cleanup_old_data():
#     current_time = time.time()
#     old_ids = [tid for tid, data in face_tracker.items() if current_time - data["last_seen"] > 5.0]
    
#     for tid in old_ids:
#         del face_tracker[tid]
#         if tid in processing_faces:
#             print(f"[CLEANUP] Removing stale track {tid}")
#             processing_faces.remove(tid)

# # -------------------------------
# # Redis subscriber for reload events
# # -------------------------------
# def listen_for_reload():
#     r = redis.Redis(host="localhost", port=6379, db=0)
#     pubsub = r.pubsub()
#     pubsub.subscribe("reload_index_channel")
#     print("[INFO] Listening for reload events...")
#     for message in pubsub.listen():
#         if message["type"] == "message":
#             print("[INFO] Reload event received!")
#             load_index()  # safely reload FAISS + metadata

# # -------------------------------
# # Recognition thread
# # -------------------------------
# # def recognize_face_thread():
# #     while True:
# #         item = recognition_queue.get()
# #         if item is None:
# #             break
# #         track_id, face_img = item
# #         try:
# #             student_folders = get_all_student_folders()

# #             person_name = None  # start with no decision
# #             best_identity = None
# #             best_distance = 1.0

# #             for folder in student_folders:
# #                 result = DeepFace.find(
# #                     face_img, db_path=folder, model_name="ArcFace",
# #                     distance_metric="cosine", detector_backend="skip",
# #                     enforce_detection=False, silent=True
# #                 )

# #                 if isinstance(result, list) and len(result) > 0:
# #                     df = result[0]
# #                     if not df.empty:
# #                         match = df.loc[df['distance'].idxmin()]
# #                         if match['distance'] < best_distance and match['distance'] < EMBEDDING_THRESHOLD:
# #                             best_distance = match['distance']
# #                             best_identity = os.path.basename(os.path.dirname(match["identity"]))

# #             if best_identity:
# #                 confidence = 1.0 - best_distance  # higher = better
# #                 person_name = best_identity
# #                 print(f"[RECOGNITION] Track {track_id} recognized as {person_name} (conf={confidence:.2f})")
# #             else:
# #                 person_name = "Unknown"

# #             with tracker_lock:
# #                 if track_id in face_tracker:
# #                     face_tracker[track_id]["rec_history"].append((person_name, confidence))


# #         except Exception as e:
# #             print(f"Error in recognition for {track_id}: {e}")
# #         finally:
# #             with tracker_lock:
# #                 if track_id in processing_faces:
# #                     processing_faces.remove(track_id)
# #             recognition_queue.task_done()

            
# ##IF FAISS IS USED
# def recognize_face_thread():
#     while True:
#         item = recognition_queue.get()
#         if item is None:
#             break
#         track_id, face_img = item
#         try:
#             result = recognize_face(face_img, threshold=EMBEDDING_THRESHOLD)

#             person_name = result.get("studentId", "Unknown")
#             confidence = result.get("similarity", 0.0)

#             with tracker_lock:
#                 if track_id in face_tracker:
#                     #if person_name != "Unknown":
#                     face_tracker[track_id]["rec_history"].append((person_name, confidence))

#             if person_name != "Unknown":
#                 print(f"[RECOGNITION] Track {track_id} recognized as {person_name} (conf={confidence:.2f})")
#             else:
#                 print(f"[RECOGNITION] Track {track_id} recognized as Unknown")


#         except Exception as e:
#             print(f"Error in recognition for {track_id}: {e}")
#         finally:
#             with tracker_lock:
#                 if track_id in processing_faces:
#                     processing_faces.remove(track_id)
#             recognition_queue.task_done()



# # -------------------------------
# # Main execution
# # -------------------------------
# if __name__ == "__main__":
#     load_index() # Load FAISS index if using FAISS

#     print("[INFO] Face detector ready.")

#     threading.Thread(target=recognize_face_thread, daemon=True).start()
#     threading.Thread(target=listen_for_reload, daemon=True).start()
#     face_detector = YOLO("yolov8s-face-lindevs.pt")
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         enhanced_frame = enhance_low_light(frame.copy())
#         frame_count += 1
#         should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

#         # Use YOLO + ByteTrack
#         results = face_detector.track(
#             enhanced_frame, conf=YOLO_CONF_THRESHOLD, persist=True, verbose=False, tracker="bytetrack.yaml"
#         )

#         if results[0].boxes.id is not None:
#             for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id.int().tolist()):
#                 x1, y1, x2, y2 = map(int, box)
#                 tid = f"person_{track_id}"

#                 # Register new track
#                 if tid not in face_tracker:
#                     print(f"[NEW TRACK] {tid} started")
#                     face_tracker[tid] = {
#                         "last_seen": time.time(),
#                         "rec_history": deque(maxlen=REC_HISTORY_LIMIT)
#                     }
#                 else:
#                     face_tracker[tid]["last_seen"] = time.time()


#                 if should_process:
#                     face_img = None
#                     h, w = enhanced_frame.shape[:2]
#                     x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
#                     if x2 > x1 and y2 > y1:
#                         face_img = enhanced_frame[y1:y2, x1:x2]

#                     if face_img is not None and face_img.size > 0:
#                         queue_face_for_recognition(tid, face_img)

#                 # Voting logic
#                 # name_to_display = "Scanning..."
#                 # color = (0, 0, 255)
#                 # rec_history = face_tracker[tid]["rec_history"]


#                 # with tracker_lock:
#                 #     rec_history = face_tracker[tid]["rec_history"]

#                 # if rec_history:
#                 #     names = [n for n, _ in rec_history]
#                 #     most_common = max(set(names), key=names.count)
#                 #     avg_conf = np.mean([c for n, c in rec_history if n == most_common])

#                 #     if names.count(most_common) >= CONFIRMATION_THRESHOLD and avg_conf >= 0.4:
#                 #         if most_common != "Unknown":
#                 #             name_to_display = f"{most_common} ({avg_conf:.2f})"
#                 #             color = (0, 255, 0)
#                 #             print(f"[CONFIRMED] {tid} → {most_common} (conf={avg_conf:.2f})")
#                 #         else:
#                 #             name_to_display = "Unknown"
#                 #             print(f"[CONFIRMED] {tid} → Unknown")

#                 # cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 2)
#                 # cv2.putText(enhanced_frame, name_to_display, (x1, y1 - 10),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#                 # cv2.putText(enhanced_frame, tid, (x1, y2 + 15),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


#                 name_to_display = "Scanning..."
#                 color = (0, 255, 255) # Yellow for "Scanning..."
#                 rec_history = face_tracker[tid]["rec_history"]

#                 if rec_history:
#                     names = [n for n, _ in rec_history]
#                     most_common = max(set(names), key=names.count)

#                     # Check if we have enough recognitions to make a decision
#                     if names.count(most_common) >= CONFIRMATION_THRESHOLD:
#                         if most_common == "Unknown":
#                             name_to_display = "Unknown"
#                             color = (0, 0, 255) # Red for "Unknown"
#                             # Optional: a print statement that only triggers once
#                             # if face_tracker[tid].get("status") != "confirmed_unknown":
#                             #     print(f"[CONFIRMED] {tid} -> Unknown")
#                             #     face_tracker[tid]["status"] = "confirmed_unknown"
#                         else:
#                             # Calculate avg confidence only for the most common (known) person
#                             avg_conf = np.mean([c for n, c in rec_history if n == most_common])
                            
#                             # Now check the confidence threshold for the known person
#                             if avg_conf >= 0.4: # Or your specific confidence threshold
#                                 name_to_display = f"{most_common} ({avg_conf:.2f})"
#                                 color = (0, 255, 0) # Green for "Confirmed"





#                                 from datetime import datetime
#                                 timestamp = datetime.now().isoformat()
                                
#                                 violation = Violation(
#                                     studentId=most_common,
#                                     cause="id card",
#                                     timestamp=timestamp,
#                                     image=f"{most_common}/{timestamp}.jpg"  # optional: save face image
#                                 )
#                                 try:
#                                     add_violation(violation)
#                                     print(f"[VIOLATION] Added for {most_common} at {timestamp}")
#                                 except Exception as e:
#                                     print(f"[VIOLATION] Could not add violation: {e}")

#                                 # Optional: a print statement that only triggers once
#                                 # if face_tracker[tid].get("status") != f"confirmed_{most_common}":
#                                 #     print(f"[CONFIRMED] {tid} -> {most_common} (conf={avg_conf:.2f})")
#                                 #     face_tracker[tid]["status"] = f"confirmed_{most_common}"

#                 # Drawing the rectangle and text
#                 cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(enhanced_frame, name_to_display, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#                 cv2.putText(enhanced_frame, tid, (x1, y2 + 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

#                 # if rec_history:
#                 #     # Separate names and confidences
#                 #     names, confs = zip(*rec_history)
                    
#                 #     # Most common predicted name
#                 #     most_common = max(set(names), key=names.count)
                    
#                 #     # Take only confidences for the most common predicted name
#                 #     relevant_confs = [c for n, c in rec_history if n == most_common]
                    
#                 #     # Temporal smoothing: average confidence over last N frames
#                 #     avg_conf = sum(relevant_confs) / len(relevant_confs)

#                 #     if names.count(most_common) >= CONFIRMATION_THRESHOLD and avg_conf >= 0.4:
#                 #         if most_common != "Unknown":
#                 #             name_to_display = f"{most_common} ({avg_conf:.2f})"
#                 #             color = (0, 255, 0)
#                 #             print(f"[CONFIRMED] {tid} → {most_common} (conf={avg_conf:.2f})")
#                 #         else:
#                 #             name_to_display = "Unknown"
#                 #     print(f"[CONFIRMED] {tid} → Unknown")


                
#         if frame_count % 30 == 0:
#             cleanup_old_data()

#         cv2.imshow("CCTV Recognition", enhanced_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     recognition_queue.put(None)
#     recognition_queue.join()




    
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import threading
from queue import Queue
import os
import time
from collections import deque
from pymongo import MongoClient
import redis
from threading import Lock
from numpy.linalg import norm
from recognition import load_index, recognize_face
from backend import Violation
from recognition import add_violation
from datetime import datetime

# -------------------------------
# MongoDB Setup
# -------------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "schoolDB"
COLLECTION_NAME = "students"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
students_col = db[COLLECTION_NAME]

def get_all_student_folders():
    """Fetch all registered student photo folders"""
    return [doc["photoFolder"] for doc in students_col.find()]

# -------------------------------
# Configuration
# -------------------------------
EMBEDDING_THRESHOLD = 0.35
REC_HISTORY_LIMIT = 10
CONFIRMATION_THRESHOLD = 3
YOLO_CONF_THRESHOLD = 0.10
ID_TIE_CONF_THRESHOLD = 0.5
MAX_QUEUE_SIZE = 8
PROCESS_EVERY_N_FRAMES = 1

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# Violation screenshot directory
VIOLATION_SCREENSHOTS_DIR = "violation_screenshots/"
os.makedirs(VIOLATION_SCREENSHOTS_DIR, exist_ok=True)

# -------------------------------
# Global variables
# -------------------------------
processing_faces = set()
recognition_queue = Queue()
tracker_lock = Lock()

# tracker: track_id -> {"last_seen": float, "rec_history": deque, "violations_logged": set}
face_tracker = {}
frame_count = 0

# -------------------------------
# Helper functions
# -------------------------------
def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def queue_face_for_recognition(track_id, face_img):
    if track_id in processing_faces:
        return
    if recognition_queue.qsize() < MAX_QUEUE_SIZE:
        processing_faces.add(track_id)
        recognition_queue.put((track_id, face_img.copy()))

def cleanup_old_data():
    current_time = time.time()
    old_ids = [tid for tid, data in face_tracker.items() if current_time - data["last_seen"] > 5.0]
    
    for tid in old_ids:
        del face_tracker[tid]
        if tid in processing_faces:
            print(f"[CLEANUP] Removing stale track {tid}")
            processing_faces.remove(tid)

# def save_violation_screenshot(frame, person_name, violation_type):
#     """Save violation screenshot with timestamp"""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{person_name}_{violation_type}_{timestamp}.jpg"
#     filepath = os.path.join(VIOLATION_SCREENSHOTS_DIR, filename)
    
#     # Create subfolder for person if needed
#     person_folder = os.path.join(VIOLATION_SCREENSHOTS_DIR, person_name)
#     os.makedirs(person_folder, exist_ok=True)
#     filepath = os.path.join(person_folder, f"{violation_type}_{timestamp}.jpg")
    
#     cv2.imwrite(filepath, frame)
#     print(f"[SCREENSHOT] Saved violation image: {filepath}")
#     return filepath

def save_violation_screenshot(frame, person_name, violation_type):
    """Save violation screenshot with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create subfolder for person if needed
    person_folder = os.path.join(VIOLATION_SCREENSHOTS_DIR, person_name)
    os.makedirs(person_folder, exist_ok=True)
    
    filename = f"{violation_type}_{timestamp}.jpg"
    filepath = os.path.join(person_folder, filename)
    
    cv2.imwrite(filepath, frame)
    print(f"[SCREENSHOT] Saved violation image: {filepath}")
    
    # Return relative path for URL access
    return f"{person_name}/{filename}"

def check_id_tie_presence(id_tie_results, face_box):
    x1, y1, x2, y2 = face_box

    def IoU(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / (boxAArea + boxBArea - interArea + 1e-5)

    has_id = False
    has_tie = False

    for box, cls in zip(id_tie_results[0].boxes.xyxy, id_tie_results[0].boxes.cls):
        obj_box = list(map(int, box))
        class_name = id_tie_results[0].names[int(cls)].lower()

        # if object overlaps lower part of body, accept it
        if IoU(face_box, obj_box) < 0.1:
            # ignore objects that overlap with face (ID shouldn't overlap face)
            pass

        # Check label names
        if "id" in class_name or "card" in class_name:
            has_id = True

        if "tie" in class_name or "neck" in class_name:
            has_tie = True

    return {"has_id": has_id, "has_tie": has_tie}


# -------------------------------
# Redis subscriber for reload events
# -------------------------------
def listen_for_reload():
    r = redis.Redis(host="localhost", port=6379, db=0)
    pubsub = r.pubsub()
    pubsub.subscribe("reload_index_channel")
    print("[INFO] Listening for reload events...")
    for message in pubsub.listen():
        if message["type"] == "message":
            print("[INFO] Reload event received!")
            load_index()

def recognize_face_thread():
    while True:
        item = recognition_queue.get()
        if item is None:
            break
        track_id, face_img = item
        try:
            result = recognize_face(face_img, threshold=EMBEDDING_THRESHOLD)
            person_name = result.get("studentId", "Unknown")
            confidence = result.get("similarity", 0.0)

            with tracker_lock:
                if track_id in face_tracker:
                    face_tracker[track_id]["rec_history"].append((person_name, confidence))

            if person_name != "Unknown":
                print(f"[RECOGNITION] Track {track_id} recognized as {person_name} (conf={confidence:.2f})")
            else:
                print(f"[RECOGNITION] Track {track_id} recognized as Unknown")

        except Exception as e:
            print(f"Error in recognition for {track_id}: {e}")
        finally:
            with tracker_lock:
                if track_id in processing_faces:
                    processing_faces.remove(track_id)
            recognition_queue.task_done()

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    load_index()
    print("[INFO] Face detector ready.")

    # Load both YOLO models
    face_detector = YOLO("yolov8s-face-lindevs.pt")
    id_tie_detector = YOLO('images_dataset/runs/detect/train5/weights/best.pt')
    print("[INFO] ID/Tie detector loaded.")

    threading.Thread(target=recognize_face_thread, daemon=True).start()
    threading.Thread(target=listen_for_reload, daemon=True).start()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # width
    cap.set(4, 720)   # height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced_frame = enhance_low_light(frame.copy())
        frame_count += 1
        should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

        # Run BOTH YOLO models on the same frame
        face_results = face_detector.track(
            enhanced_frame, conf=YOLO_CONF_THRESHOLD, persist=True, verbose=False, tracker="bytetrack.yaml"
        )
        
        id_tie_results = id_tie_detector(enhanced_frame, conf=ID_TIE_CONF_THRESHOLD, verbose=False)

        # Draw ID/Tie detections (optional visualization)
        if id_tie_results[0].boxes is not None:
            for box, cls in zip(id_tie_results[0].boxes.xyxy, id_tie_results[0].boxes.cls):
                obj_x1, obj_y1, obj_x2, obj_y2 = map(int, box)
                class_name = id_tie_results[0].names[int(cls)]
                cv2.rectangle(enhanced_frame, (obj_x1, obj_y1), (obj_x2, obj_y2), (255, 0, 255), 2)
                cv2.putText(enhanced_frame, class_name, (obj_x1, obj_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        if face_results[0].boxes.id is not None:
            for box, track_id in zip(face_results[0].boxes.xyxy, face_results[0].boxes.id.int().tolist()):
                x1, y1, x2, y2 = map(int, box)
                tid = f"person_{track_id}"

                # Register new track
                if tid not in face_tracker:
                    print(f"[NEW TRACK] {tid} started")
                    # face_tracker[tid] = {
                    #     "last_seen": time.time(),
                    #     "rec_history": deque(maxlen=REC_HISTORY_LIMIT),
                    #     "violations_logged": set()  # Track which violations have been logged
                    # }

                    face_tracker[tid] = {
                        "last_seen": time.time(),
                        "rec_history": deque(maxlen=REC_HISTORY_LIMIT),
                        "violations_logged": set(),
                        "id_history": deque(maxlen=REC_HISTORY_LIMIT),
                        "tie_history": deque(maxlen=REC_HISTORY_LIMIT),
                        "violation_counters": { "NO_ID": 0, "NO_TIE": 0 }
                    }

                else:
                    face_tracker[tid]["last_seen"] = time.time()

                if should_process:
                    face_img = None
                    h, w = enhanced_frame.shape[:2]
                    x1_clip, y1_clip = max(0, x1), max(0, y1)
                    x2_clip, y2_clip = min(w, x2), min(h, y2)
                    
                    if x2_clip > x1_clip and y2_clip > y1_clip:
                        face_img = enhanced_frame[y1_clip:y2_clip, x1_clip:x2_clip]

                    if face_img is not None and face_img.size > 0:
                        queue_face_for_recognition(tid, face_img)

                name_to_display = "Scanning..."
                color = (0, 255, 255)
                rec_history = face_tracker[tid]["rec_history"]

                if rec_history:
                    names = [n for n, _ in rec_history]
                    most_common = max(set(names), key=names.count)

                    if names.count(most_common) >= CONFIRMATION_THRESHOLD:
                        if most_common == "Unknown":
                            name_to_display = "Unknown"
                            color = (0, 0, 255)
                        else:
                            avg_conf = np.mean([c for n, c in rec_history if n == most_common])
                            
                            if avg_conf >= 0.4:
                                name_to_display = f"{most_common} ({avg_conf:.2f})"
                                
                                # Check ID and Tie presence
                                # id_tie_status = check_id_tie_presence(id_tie_results, [x1, y1, x2, y2])
                                
                                # violations = []
                                # if not id_tie_status["has_id"]:
                                #     violations.append("NO_ID")
                                # if not id_tie_status["has_tie"]:
                                #     violations.append("NO_TIE")
                                
                                id_tie_status = check_id_tie_presence(id_tie_results, [x1, y1, x2, y2])

                                # store history
                                face_tracker[tid]["id_history"].append(1 if id_tie_status["has_id"] else 0)
                                face_tracker[tid]["tie_history"].append(1 if id_tie_status["has_tie"] else 0)

                                # temporal majority: missing only if mostly 0
                                id_present = sum(face_tracker[tid]["id_history"]) > len(face_tracker[tid]["id_history"]) // 2
                                tie_present = sum(face_tracker[tid]["tie_history"]) > len(face_tracker[tid]["tie_history"]) // 2

                                violations = []
                                if not id_present:
                                    violations.append("NO_ID")
                                if not tie_present:
                                    violations.append("NO_TIE")

                                # Update counters
                                for vtype in ["NO_ID", "NO_TIE"]:
                                    if vtype in violations:
                                        face_tracker[tid]["violation_counters"][vtype] += 1
                                    else:
                                        # Decay or reset counter if violation not present in this frame
                                        face_tracker[tid]["violation_counters"][vtype] = max(
                                            0, face_tracker[tid]["violation_counters"][vtype] - 1)

                                # Log violation only if counter > threshold (e.g., 3 times)
                                VIOLATION_LOG_THRESHOLD = 15

                                for violation_type in violations:
                                    if face_tracker[tid]["violation_counters"][violation_type] >= VIOLATION_LOG_THRESHOLD:
                                        per_type_key = f"{most_common}_{violation_type}"
                                        if per_type_key not in face_tracker[tid]["violations_logged"]:
                                            # Log violation as before, add to violations_logged to prevent duplicates
                                            # Log violations if any
                                            # mapping for readable cause names
                                            VIOLATION_CAUSE_MAP = {
                                                "NO_ID": "id card",
                                                "NO_TIE": "tie",
                                                # add future mapping here, e.g. "NO_MASK": "mask"
                                            }

                                            if violations:
                                                color = (0, 0, 255)  # Red for violation

                                                # Save ONE screenshot for this detection event (filename can include combined types)
                                                timestamp = datetime.now().isoformat()
                                                screenshot_path = save_violation_screenshot(frame, most_common, "_".join(violations))

                                                # Ensure face_tracker structure exists
                                                if "violations_logged" not in face_tracker[tid]:
                                                    face_tracker[tid]["violations_logged"] = set()

                                                if len(violations) == 1:
                                                    # single violation (ID only or tie only)
                                                    violation_type = violations[0]  # extract the single violation string
                                                    per_type_key = f"{most_common}_{violation_type}"
                                                   

                                                    if per_type_key not in face_tracker[tid]["violations_logged"]:
                                                        cause = [VIOLATION_CAUSE_MAP.get(violation_type, violation_type)]
                                                        violation = Violation(
                                                            studentId=most_common,
                                                            cause=cause,
                                                            timestamp=timestamp,
                                                            image=screenshot_path
                                                        )
                                                        add_violation(violation)
                                                        face_tracker[tid]["violations_logged"].add(per_type_key)
                                                else:
                                                    # both missing → one combined record
                                                    per_type_key = f"{most_common}_COMBINED_NO_ID_NO_TIE"
                                                    if per_type_key not in face_tracker[tid]["violations_logged"]:
                                                        cause = [VIOLATION_CAUSE_MAP.get(v, v) for v in violations]
                                                        violation = Violation(
                                                            studentId=most_common,
                                                            cause=cause,  # ["id card", "tie"]
                                                            timestamp=timestamp,
                                                            image=screenshot_path
                                                        )
                                                        add_violation(violation)
                                                        face_tracker[tid]["violations_logged"].add(per_type_key)

                                                name_to_display = f"{most_common} - {', '.join(violations)}"


                                            else:
                                                color = (0, 255, 0)  # Green for no violations
                                                name_to_display = f"{most_common} ({avg_conf:.2f})"



                # Drawing
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(enhanced_frame, name_to_display, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(enhanced_frame, tid, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if frame_count % 30 == 0:
            cleanup_old_data()

        cv2.imshow("CCTV Recognition", enhanced_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognition_queue.put(None)
    recognition_queue.join()
