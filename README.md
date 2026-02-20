# Face Detection Attendance Project

A real-time face recognition-based attendance system built using Python, OpenCV, and the face_recognition library.

**Project Description**

This project detects and recognizes faces using a webcam and automatically records attendance in a CSV file.

The system:

-Detects faces in real-time
-Encodes faces into 128-d embeddings
-Recognizes users from stored encodings
-Logs attendance with timestamp
-Provides voice confirmation when attendance is recorded

**Improvement & Optimization**

The project was originally implemented using KNN for face recognition.

It was later improved by replacing KNN with:

face_distance + threshold matching

Why the improvement?

-More accurate matching

-Better performance for small datasets

-Reduced misclassification between similar faces

-Simpler and more efficient decision logic

-This optimized version provides better balance between speed and accuracy on CPU-based systems.


**How to Run**
Install dependencies->pip install opencv-python face_recognition numpy
Add a new user->python add_faces.py
Run the system->python test.py
Press:
o → Record attendance
q → Quit


