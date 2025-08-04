import cv2
import numpy as np
import pickle
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3

# Initialize TTS engine (cross-platform)
engine = pyttsx3.init()
engine.say("Initializing face recognition voting system.")
engine.runAndWait()

# Load data
if not os.path.exists('data/'):
    engine.say("No data folder found. Please add face data first.")
    engine.runAndWait()
    exit()

# Load profiles (voter info)
if not os.path.exists('data/profile.pkl') or not os.path.exists('data/faces_data.pkl'):
    engine.say("Required data files are missing.")
    engine.runAndWait()
    exit()

with open('data/profile.pkl', 'rb') as f:
    profile_list = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    face_data = pickle.load(f)

if len(profile_list) != len(face_data):
    print("Mismatch in data size.")
    exit()

# Prepare voter labels
labels = [profile['Voter ID'] for profile in profile_list]

# Train classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(face_data, labels)

# Load Haar Cascade for face detection
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start webcam
video_capture = cv2.VideoCapture(0)

# Columns for vote log
col_names = ['Voter ID', 'Name', 'Constituency', 'Vote', 'Time', 'Date']

# Voting loop
voted_users = set()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_section = cv2.resize(crop_img, (100, 100))
        face_flat = face_section.flatten().reshape(1, -1)

        voter_id_pred = knn.predict(face_flat)[0]

        # Avoid re-voting
        if voter_id_pred in voted_users:
            cv2.putText(frame, "Already Voted", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            continue

        # Get voter profile info
        try:
            idx = labels.index(voter_id_pred)
            profile = profile_list[idx]
            name = profile['Name']
            constituency = profile['Constituency']
        except:
            continue  # ID not found, skip

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Mark as voted
        voted_users.add(voter_id_pred)

        # Save vote
        if 'votes.csv' not in os.listdir('data/'):
            with open('data/votes.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(col_names)
        
        with open('data/votes.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                voter_id_pred,
                name,
                constituency,
                'YES',
                datetime.now().strftime("%H:%M:%S"),
                datetime.now().strftime("%Y-%m-%d")
            ])

        # Say thank you
        engine.say(f"Thank you {name}, your vote has been recorded.")
        engine.runAndWait()

        time.sleep(3)  # Pause for user feedback
        break  # Optional: stop after one vote

    cv2.imshow('Voting Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
