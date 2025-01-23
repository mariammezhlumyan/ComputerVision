import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(1)

student_images_path = "Students/"

known_face_encodings = []
known_face_names = []

for filename in os.listdir(student_images_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')): 
        name = os.path.splitext(filename)[0] 
        known_face_names.append(name)

        image_path = os.path.join(student_images_path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            known_face_encodings.append(encoding[0])
        else:
            print(f"Warning: No face found in {filename}")

students = known_face_names.copy()

now = datetime.now()
current_date = now.strftime("%m-%d-%Y")
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])

while True:
    _, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    face_encodings = []
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = ""
        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        if name in students:
            students.remove(name)
            print(f"{name} marked as present.")
            current_time = datetime.now().strftime("%H:%M:%S")
            lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

csv_files = [file for file in os.listdir() if file.endswith('.csv')]

all_students = set()
for file in csv_files:
    df = pd.read_csv(file)
    all_students.update(df['Name']) 

all_students = sorted(list(all_students))

attendance_df = pd.DataFrame({'Name': all_students})

for file in csv_files:
    date = os.path.splitext(file)[0]
    
    df = pd.read_csv(file)
    present_students = df['Name'].tolist()
    
    attendance_df[date] = attendance_df['Name'].apply(lambda x: 'Present' if x in present_students else 'Absent')

attendance_df.to_csv('Attendance.csv', index=False)

print("Attendance has been saved as 'Attendance.csv'.")