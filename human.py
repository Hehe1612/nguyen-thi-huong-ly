import cv2
import numpy as np
import face_recognition
import os
import sys
import math


# Reliability of identification
def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = 1.0 - face_match_threshold
    linear_value = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_value * 100, 2)) + "%"
    else:
        value = (
            linear_value
            + ((1.0 - linear_value) * math.pow((linear_value - 0.5) ** 2, 0.22))
        ) * 100
        return str(round(value, 2)) + "%"


class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.encode_faces()

    # Nhận diện khuôn mặt từ hình ảnh
    def encode_faces(self):
        image_paths = [
            r"C:\MEARCHING LEARNING\bearbrick.jpg",
            r"C:\MEARCHING LEARNING\anh_nhat.jpg",
            r"C:\MEARCHING LEARNING\huong_ly.jpg",
        ]

        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Không tìm thấy file: {image_path}")
                continue

            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)

            if face_encoding:  # Kiểm tra xem có tìm thấy khuôn mặt hay không
                self.known_face_encodings.append(face_encoding[0])
                self.known_face_names.append(
                    os.path.basename(image_path)
                )  # Lưu tên file làm tên người

        print("Khuôn mặt đã mã hóa:", self.known_face_names)

    # Chạy nhận diện khuôn mặt real-time
    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Không tìm thấy nguồn video..!")

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations
                )
                self.face_names = []

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding
                    )
                    name = "Unknown"
                    confidence = "Unknown"
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    best_match_index = (
                        np.argmin(face_distances) if len(face_distances) > 0 else -1
                    )

                    if best_match_index != -1 and matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f"{name} ({confidence})")

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(
                self.face_locations, self.face_names
            ):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1
                )
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FaceRecognition()
    fr.run_recognition()
