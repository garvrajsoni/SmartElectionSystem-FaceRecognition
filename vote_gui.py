import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import pickle
import numpy as np
from datetime import datetime
import csv
import os
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3

# Text-to-Speech Init
engine = pyttsx3.init()

# Load Haar Cascade
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load KNN Model
with open('data/profile.pkl', 'rb') as f:
    profile_list = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    face_data = pickle.load(f)

labels = [profile['Voter ID'] for profile in profile_list]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(face_data, labels)

# Track votes to avoid repeat
voted_users = set()

# GUI App
class SmartVotingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Voting System")
        self.root.geometry("1050x600")
        self.root.resizable(False, False)

        self.video_label = tk.Label(root)
        self.video_label.place(x=20, y=70, width=600, height=450)

        self.vote = None
        self.current_voter = None

        # Buttons
        self.create_buttons()

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def create_buttons(self):
        options = [
            ("BJP", "1", "orange"),
            ("CONGRESS", "2", "green"),
            ("AAP", "3", "blue"),
            ("NOTA", "4", "red")
        ]
        for idx, (party, key, color) in enumerate(options):
            btn = tk.Button(self.root, text=f'Press "{key}" for "{party}"',
                            command=lambda p=party: self.record_vote(p),
                            bg=color, fg="white", font=("Helvetica", 14), width=22)
            btn.place(x=680, y=120 + idx * 80)

        title = tk.Label(self.root, text="SMART VOTING SYSTEM", font=("Helvetica", 20, "bold"), fg="skyblue")
        title.place(x=360, y=20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            face = cv2.resize(crop, (100, 100)).flatten().reshape(1, -1)

            voter_id = knn.predict(face)[0]

            if voter_id in voted_users:
                label = "Already Voted"
            else:
                try:
                    idx = labels.index(voter_id)
                    profile = profile_list[idx]
                    name = profile["Name"]
                    label = f"{name}"
                    self.current_voter = profile
                except:
                    label = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # handle only first face

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def record_vote(self, party_name):
        if not self.current_voter:
            messagebox.showwarning("No Face", "No valid voter detected.")
            return

        voter_id = self.current_voter["Voter ID"]
        name = self.current_voter["Name"]
        constituency = self.current_voter["Constituency"]

        if voter_id in voted_users:
            messagebox.showinfo("Duplicate", f"{name} has already voted.")
            return

        voted_users.add(voter_id)

        vote_data = [voter_id, name, constituency, party_name,
                     datetime.now().strftime("%H:%M:%S"),
                     datetime.now().strftime("%Y-%m-%d")]

        # Save to CSV
        if not os.path.exists('data/votes.csv'):
            with open('data/votes.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Voter ID', 'Name', 'Constituency', 'Vote', 'Time', 'Date'])

        with open('data/votes.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(vote_data)

        engine.say(f"Thank you {name}. Your vote for {party_name} has been recorded.")
        engine.runAndWait()
        messagebox.showinfo("Vote Recorded", f"Vote recorded for {name} - {party_name} ✅")

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = SmartVotingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
