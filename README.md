# Face Detection Attendance Project

**Project Goal:**  
Automatically detect faces using your webcam and record attendance in a CSV file.

## How it Works
1. **Add your face**
   - Run `add_faces.py`
   - Capture multiple images of yourself.
2. **Take attendance**
   - Run `test.py`
   - The system detects your face and saves your name and timestamp in `Attendance/Attendance_DD-MM-YYYY.csv`.
3. **Check attendance**
   - Open CSV files in `Attendance/` folder to see who attended.

## Requirements
pip install -r requirements.txt
## Note
- This repository **does not include personal photos or generated attendance CSV files**.
- Each user should run `add_faces.py` to register their face and then use `test.py` to mark attendance.

