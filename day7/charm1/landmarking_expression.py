import cv2
import numpy as np
import dlib
import os

detector = dlib.get_frontal_face_detector()

# from this predictor we will get the points to plot on face
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

mood = input("Enter your mood : ")

frames = []
outputs = []


while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # mouth expression
        expression = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[17:]])

    if ret:
        cv2.imshow("My screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord('c'):
        frames.append(expression.flatten())
        outputs.append([mood])


X = np.array(frames)
y = np.array(outputs)

# stacking horizontally so that 0th loc will have output and rest will have input(features)
data = np.hstack([y, X])

f_name = "face_mood.npy"

# adding extra data to the same file if it exists
if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)

cap.release()
cv2.destroyAllWindows()