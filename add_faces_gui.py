import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import pickle
import os
from PIL import Image, ImageTk

# Ensure data folder exists
os.makedirs('data/', exist_ok=True)

# Load Haar Cascade
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# GUI Class
class FaceRegistrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Register Voter - Smart Voting System")
        self.root.geometry("1050x600")
        self.root.resizable(False, False)

        self.cap = None
        self.face_data = []
        self.frame_count = 0

        self.build_form()
        self.build_video_area()

    def build_form(self):
        tk.Label(self.root, text="Enter Voter Details", font=("Helvetica", 16, "bold")).place(x=700, y=30)

        tk.Label(self.root, text="Voter ID (12-digits):").place(x=700, y=80)
        self.voter_id_entry = tk.Entry(self.root)
        self.voter_id_entry.place(x=700, y=105)

        tk.Label(self.root, text="Name:").place(x=700, y=140)
        self.name_entry = tk.Entry(self.root)
        self.name_entry.place(x=700, y=165)

        tk.Label(self.root, text="Constituency:").place(x=700, y=200)
        self.const_entry = tk.Entry(self.root)
        self.const_entry.place(x=700, y=225)

        self.capture_btn = tk.Button(self.root, text="Capture Faces", bg="blue", fg="white", font=("Helvetica", 12),
                                     command=self.start_capture)
        self.capture_btn.place(x=700, y=270)

    def build_video_area(self):
        self.video_label = tk.Label(self.root)
        self.video_label.place(x=30, y=50, width=600, height=450)

    def start_capture(self):
        self.face_data = []
        self.frame_count = 0
        self.voter_id = self.voter_id_entry.get()
        self.name = self.name_entry.get()
        self.constituency = self.const_entry.get()

        # Validations
        if not self.voter_id.isdigit() or len(self.voter_id) != 12:
            messagebox.showerror("Error", "Voter ID must be a 12-digit number.")
            return
        if not self.name.replace(" ", "").isalpha() or not self.constituency.replace(" ", "").isalpha():
            messagebox.showerror("Error", "Name and Constituency must only contain letters.")
            return

        self.profile = {
            'Voter ID': self.voter_id,
            'Name': self.name,
            'Constituency': self.constituency
        }

        self.cap = cv2.VideoCapture(0)
        self.capture_faces()

    def capture_faces(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            face_section = cv2.resize(crop, (100, 100))

            if self.frame_count % 2 == 0 and len(self.face_data) < 50:
                self.face_data.append(face_section)

            self.frame_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Captured: {len(self.face_data)}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break

        if len(self.face_data) >= 50:
            self.cap.release()
            self.save_data()
            messagebox.showinfo("Success", f"Faces captured for {self.name}.")
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.capture_faces)

    def save_data(self):
        self.face_data = np.asarray(self.face_data).reshape(len(self.face_data), -1)
        profile_list = [self.profile] * len(self.face_data)

        # Save profile
        if 'profile.pkl' not in os.listdir('data/'):
            with open('data/profile.pkl', 'wb') as f:
                pickle.dump(profile_list, f)
        else:
            with open('data/profile.pkl', 'rb') as f:
                existing_profiles = pickle.load(f)
                existing_profiles.extend(profile_list)
            with open('data/profile.pkl', 'wb') as f:
                pickle.dump(existing_profiles, f)

        # Save face data
        if 'faces_data.pkl' not in os.listdir('data/'):
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(self.face_data, f)
        else:
            with open('data/faces_data.pkl', 'rb') as f:
                existing_data = pickle.load(f)
                combined_data = np.concatenate((existing_data, self.face_data), axis=0)
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(combined_data, f)


# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRegistrationApp(root)
    root.mainloop()
