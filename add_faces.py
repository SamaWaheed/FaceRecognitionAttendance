import cv2
import pickle
import os
import numpy as np

facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
name = input("Enter your name: ")

faces_data_path = 'data/faces_data.pkl'
names_path = 'data/names.pkl'

if os.path.exists(faces_data_path):
    with open(faces_data_path, 'rb') as f:
        faces_data = pickle.load(f)
    with open(names_path, 'rb') as f:
        labels = pickle.load(f)
else:
    faces_data = []
    labels = []

count = 0
total_images = 25

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (23,25)).flatten()
        if count < total_images:
            faces_data.append(resized_img)
            labels.append(name)
            count += 1
            cv2.putText(frame, f"Captured: {count}/{total_images}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Add Face", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or count >= total_images:
        break

video.release()
cv2.destroyAllWindows()
with open(faces_data_path, 'wb') as f:
    pickle.dump(np.array(faces_data), f)
with open(names_path, 'wb') as f:
    pickle.dump(labels, f)
print(f"{count} face images saved successfully for {name}!")
