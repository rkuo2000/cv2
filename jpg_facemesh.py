import sys
import cv2
import mediapipe as mp

filename = sys.argv[1]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

image = cv2.imread(filename)
# Convert the BGR image to RGB before processing.
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Print and draw face mesh landmarks on the image.
annotated_image = image

for face_landmarks in results.multi_face_landmarks:
    print('face_landmarks:', face_landmarks)
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)
		
cv2.imwrite('facemesh.png', image)
cv2.imshow('FaceMesh', image)
face_mesh.close()

cv2.waitKey(0)
cv2.destroyAllWindows()

