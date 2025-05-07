import face_recognition as fr
import cv2
import numpy as np
import os
from threading import Thread

train_folder = r"C:\MEARCHING LEARNING\train"
tolerance = 1.0
frame_resize_scale = 0.48  # Balance accuracy and speed
font = cv2.FONT_HERSHEY_DUPLEX
detection_interval = 5  # Detect max 5 face human

# Load training images
known_names = []
spoken_names = set()  # To track names that have been spoken
known_encodings = []

for filename in os.listdir(train_folder):
    path = os.path.join(train_folder, filename)
    image = fr.load_image_file(path)
    encodings = fr.face_encodings(image)
    known_encodings.append(encodings[0])
    name = os.path.splitext(filename)[0].title()
    known_names.append(name)

# Np has speed faster
known_encodings = np.array(known_encodings)

# Start webcam face detection
video_capture = cv2.VideoCapture(0)
detecting = True  # Toggle detection on/off
frame_count = 0
face_locations = []
face_encodings = []

# Detect face in extension
def detect_and_encode(frame):
    global face_locations, face_encodings
    small_frame = cv2.resize(frame, (0, 0), fx=frame_resize_scale, fy=frame_resize_scale)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_small_frame, model="hog")
    face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Each detect face human in  detection_interval frame
    if detecting and frame_count % detection_interval == 0:
        thread = Thread(target=detect_and_encode, args=(frame,))
        thread.start()
        thread.join()  # Wait frame completed

    # Process all detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = fr.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        name = "Unknown"

        if distances[best_match_index] < tolerance:
            name = known_names[best_match_index]

        # Scale back up to original frame size
        scale = int(1 / frame_resize_scale)
        top, right, bottom, left = top * scale, right * scale, bottom * scale, left * scale

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 8), font, 0.7, (255, 255, 255), 1)

    # Display detection status
    mode_text = "Detecting: ON" if detecting else "Detecting: OFF"
    cv2.putText(frame, mode_text, (10, 30), font, 0.7, (0, 0, 255), 2)

    cv2.imshow("🔍 Real-time Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit
        break
    elif key == ord("f"):  # Toggle detection
        detecting = not detecting

    frame_count += 1

video_capture.release()
cv2.destroyAllWindows()
