# pip install google.generativeai

import google.generativeai as genai
import cv2
import os
from PIL import Image
from gtts import gTTS

GOOGLE_API_KEY="get it from Google AI Studio" ## https://aistudio.google.com/app/apikey
genai.configure(api_key=GOOGLE_API_KEY)
        
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    #print(frame.shape)
    #frame = cv2.flip(frame, 1) # 0: vertical flip, 1: horizontal flip
    cv2.imshow('Camera', frame)
    #cv2.imwrite('cam.jpg',frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break        
    elif cv2.waitKey(10) & 0xFF == ord(' '):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)      
        #img = Image.open('cam.jpg')
        prompt = "照片中是什麼食物, 請簡單回答!"
        model = genai.GenerativeModel("gemini-1.5-flash")

        result = model.generate_content( [prompt , img] )
        print(result.text)
        
        # Text-to-Speech
        tts = gTTS(result.text,lang="zh-TW")
        tts.save('gTTS.mp3')
        os.system('cmdmp3 gTTS.mp3') # Windows        
        
cap.release()
cv2.destroyAllWindows()
