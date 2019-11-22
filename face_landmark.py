# Install : pip install face_recognition (dlib will be built)

# Usage: python jpg_face_landmark.py 1.jpg
import face_recognition
import cv2
import sys

# Load the jpg file into a numpy array
filename = sys.argv[1]
image = face_recognition.load_image_file(filename)

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

    count=0
    for facial_feature in facial_features:
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
        for point in face_landmarks[facial_feature]:
            cv2.circle(image, point, 1, (0,255,0), -1)
            count+=1
    print('landmark count = ', count)

cv2.imshow('face landmark', image)
cv2.waitKey()
cv2.closeAllWindows()
