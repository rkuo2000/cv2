### MTCNN face detection
# pip install mtcnn
import cv2
from mtcnn.mtcnn import MTCNN

PADDING = 50

def webcam_face_recognizer():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

#    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # OpenCV CascadeClassifier
    face_detector = MTCNN()
    
    while vc.isOpened():
        _, frame = vc.read()	
        img = process_frame(frame, face_detector)   
        
        key = cv2.waitKey(100)
        cv2.imshow("preview", img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def process_frame(img, detector):
    faces = detector.detect_faces(img)
    print(faces)

    for face in faces:
        [x, y, w, h] = face['box']
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING
        img = cv2.rectangle(img,(x1, y1),(x2, y2),(255,0,0),2)
    return img

if __name__ == "__main__":
    webcam_face_recognizer()