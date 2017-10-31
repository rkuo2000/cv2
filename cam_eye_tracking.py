import face_recognition
import cv2

cap = cv2.VideoCapture(0)

while 1:
    ret, image = cap.read()

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    #print("I found {} face(s) in thisc photograph.".format(len(face_landmarks_list)))

    for face_landmarks in face_landmarks_list:
        for dot in face_landmarks['left_eye']:
            cv2.line(image, dot, dot, (0,255,0), 1)
        for dot in face_landmarks['right_eye']:
            cv2.line(image, dot, dot, (0,255,0), 1)			
        cv2.imshow('FACIAL LANDMARK', image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
		
cap.release()
cv2.destroyAllWindows()