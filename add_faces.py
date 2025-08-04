import cv2
import pickle
import os
import numpy as np

if not os.path.exists('data/'):
    os.makedirs('data/')

video_capture = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Load the pre-trained face detection model

face_data = []

i = 0

elect_no= input("Enter your Voter ID card number: ")
name = input("Enter your name: ")
constituency = input("Enter your constituency: ")



if(not elect_no.isdigit() or len(elect_no) != 12):
    print("Invalid Voter ID card number. Please enter a 12-digit number.")
    exit()
if not name or not constituency or not name.isalpha() or not constituency.isalpha():
    print("Name and constituency should not contain spaces.")
    exit()


elect_prof = {
    'Voter ID': elect_no,
    'Name': name,
    'Constituency': constituency
}

frameTotal = 50
captureAfter = 2

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break
  
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert the frame to grayscale
    faces = face_detect.detectMultiScale(gray_frame, 1.3, 5)

    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) #Draw rectangle around the face. here frame is the original image, (x,y) is the top-left corner of the rectangle, and (w,h) are the width and height of the rectangle.
        if(len(face_data)<= frameTotal) and i% captureAfter == 0:
            face_section = cv2.resize(crop_img, (100, 100))  # or any consistent size
            face_data.append(face_section)
        i += 1
        cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Video Frame', frame)

    k=cv2.waitKey(1) & 0xff #Wait for a key press

    if k == ord('q') or len(face_data) >= frameTotal:
        print("Exiting...")
        break

video_capture.release()
cv2.destroyAllWindows()

print("Total faces captured: ", len(face_data))
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1)) #Reshape the data to a 2D array where each row is a flattened image
print("Face data shape: ", face_data.shape)

elect_list = [elect_prof] * frameTotal  # repeat the full profile for each face


if 'profile.pkl' not in os.listdir('data/'):
    with open('data/profile.pkl', 'wb') as f:
        pickle.dump(elect_list, f)
else:
    with open('data/profile.pkl', 'rb') as f:
        existing_profiles = pickle.load(f)
        existing_profiles.extend(elect_list)
    with open('data/profile.pkl', 'wb') as f:
        pickle.dump(existing_profiles, f)



if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        existing_data = pickle.load(f)
        face_data = np.concatenate((existing_data, face_data), axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
