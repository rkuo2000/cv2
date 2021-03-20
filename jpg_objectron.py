import cv2
import mediapipe as mp

file = 'shoes.jpg'
#file = 'chair.jpg'

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# For static images:
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Shoe') as objectron:
#                            model_name='Chair') as objectron:
							
#  for idx, file in enumerate(file_list):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw box landmarks.
    if not results.detected_objects:
        print(f'No box landmarks detected on {file}')
#        continue
    print(f'Box landmarks of {file}:')
    annotated_image = image.copy()
    for detected_object in results.detected_objects:
        mp_drawing.draw_landmarks(
          annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
        mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                           detected_object.translation)
        cv2.imwrite('shoes_annotated.png', annotated_image)
        cv2.imshow('annotated', annotated_image)
	  
cv2.waitKey(0)
cv2.destroyAllWindows()
