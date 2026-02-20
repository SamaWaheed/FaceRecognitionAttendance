import cv2
import pickle
import os
import numpy as np

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input("Enter your name: ")

faces_data = []
count = 0
max_faces = 50

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (23,25)).flatten()

        if count < max_faces:
            faces_data.append(resized_img)
            count += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{count}/{max_faces}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Capture Faces", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or count >= max_faces:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.array(faces_data)
labels = np.array([name]*len(faces_data))  

faces_file = 'data/faces_data.pkl'
names_file = 'data/names.pkl'

if os.path.exists(faces_file):
    with open(faces_file, 'rb') as f:
        old_faces = pickle.load(f)
    faces_data = np.vstack((old_faces, faces_data))

if os.path.exists(names_file):
    with open(names_file, 'rb') as f:
        old_labels = pickle.load(f)
    labels = np.hstack((old_labels, labels))

with open(faces_file, 'wb') as f:
    pickle.dump(faces_data, f)

with open(names_file, 'wb') as f:
    pickle.dump(labels, f)

print(f"Saved {count} faces for '{name}' successfully!")
