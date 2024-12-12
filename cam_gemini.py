# pip install google.generativeai

import google.generativeai as genai
import cv2
import time
from PIL import Image

GOOGLE_API_KEY="get it from Google AI studio" ## https://aistudio.google.com/app/apikey
genai.configure(api_key=GOOGLE_API_KEY)
        
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    #print(frame.shape)
    frame = cv2.flip(frame, 1) # 0: vertical flip, 1: horizontal flip
    cv2.imshow('Camera', frame)
    #cv2.imwrite('cam.jpg',frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break        
    elif cv2.waitKey(10) & 0xFF == ord(' '):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)      
        #img = Image.open('cam.jpg')
        prompt = "請問照片中看到甚麼?"
        model = genai.GenerativeModel("gemini-1.5-flash")

        result = model.generate_content( [prompt , img] )
        print(result.text)
        
#    time.sleep(1) 
cap.release()
cv2.destroyAllWindows()
