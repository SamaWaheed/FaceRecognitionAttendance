from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    Dispatch("SAPI.SpVoice").Speak(text)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    outputs = []
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (23,25)).flatten().reshape(1,-1)
        output = knn.predict(resized_img)
        outputs.append(output[0])
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 2)
        cv2.rectangle(frame, (x,y-40), (x+w,y), (50,50,255), -1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)
    if k == ord('o'):
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%I:%M %p - %d/%m/%Y")
        file_path = "Attendance/Attendance_" + date + ".csv"
        exist = os.path.isfile(file_path)
        for name in outputs:
            attendance = [str(name), timestamp]
            if exist:
                with open(file_path, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    names_list = [row[0] for row in reader if len(row) > 0]
                if str(name) not in names_list:
                    with open(file_path, "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(attendance)
                    speak(f"Attendance Taken for {name}")
            else:
                with open(file_path, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
                speak(f"Attendance Taken for {name}")
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
