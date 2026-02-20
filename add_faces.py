import cv2
import face_recognition
import pickle
import os
import numpy as np
import time

video = cv2.VideoCapture(0)

name = input("Enter your name: ")

encodings = []
count = 0
max_images = 20
last_capture_time = 0
capture_interval = 0.7

while True:
    ret, frame = video.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb_frame)

    for face in faces:
        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        current_time = time.time()

        if count < max_images and (current_time - last_capture_time) > capture_interval:
            face_encoding = face_recognition.face_encodings(rgb_frame, [face])[0]
            encodings.append(face_encoding)
            count += 1
            last_capture_time = current_time

        cv2.putText(frame, f"{count}/{max_images}", (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) == ord('q') or count >= max_images:
        break

video.release()
cv2.destroyAllWindows()

encodings = np.array(encodings)
labels = np.array([name]*len(encodings))

enc_file = 'data/encodings.pkl'
name_file = 'data/names.pkl'

if os.path.exists(enc_file):
    with open(enc_file, 'rb') as f:
        old_encodings = pickle.load(f)
    encodings = np.vstack((old_encodings, encodings))

if os.path.exists(name_file):
    with open(name_file, 'rb') as f:
        old_names = pickle.load(f)
    labels = np.hstack((old_names, labels))

with open(enc_file, 'wb') as f:
    pickle.dump(encodings, f)

with open(name_file, 'wb') as f:
    pickle.dump(labels, f)

print(f"Saved {count} face encodings for '{name}' successfully!")
