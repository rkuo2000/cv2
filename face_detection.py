# Install: pip install mtcnn
# Usage: python face_detection.py friends.jpg
from mtcnn import MTCNN
import cv2
import sys

if len(sys.argv)>1:
    img = cv2.imread(sys.argv[1])
else:
    img = cv2.imread("friends.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = MTCNN()

faces = detector.detect_faces(rgb)

for face in faces:
    box = face['box']
    print(box)
    keypoints= face['keypoints']
    print(keypoints)
    confidence= face['confidence']
    print(confidence)
    print()
    cv2.rectangle(img, tuple(box), (0,255,0), 2)

cv2.imshow('MTCNN face detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
