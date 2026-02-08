import os, sys
import cv2
import numpy as np
import pickle
import random
import time
from datetime import datetime
import mail
import glob
from mail import*
from PIL import Image
# from openpyxl import load_workbook
import pandas as pd
import serial 
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

capture_duration = 20
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer//model.yml")

id=0
count = 0
count_ = 0
ret,img=cam.read()
start_time = time.time()
ids = '$,'

def report_send_mail(label, image_path):
    with open(image_path, 'rb') as f:
        img_data = f.read()
    fromaddr = "kavinmec92@gmail.com"
    toaddr = "kavinmec92@gmail.com"
    msg = MIMEMultipart() 
    msg['From'] = fromaddr 
    msg['To'] = toaddr 
    msg['Subject'] = "Alert"
    body = str(label)
    msg.attach(MIMEText(body, 'plain'))  # attach plain text
    image = MIMEImage(img_data, name=os.path.basename(image_path))
    msg.attach(image) # attach image
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    s.starttls() 
    s.login(fromaddr, "wrhfrmevjiqilfas") 
    text = msg.as_string() 
    s.sendmail(fromaddr, toaddr, text) 
    s.quit()
        
while( int(time.time() - start_time) < capture_duration):

    _,img=cam.read()
    font = cv2.FONT_HERSHEY_PLAIN
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.rectangle(img,(x-50,y-50),(x+w+50,y+h+50),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (conf<60):
            if id == 1 or id == 2 or id == 3:
                cv2.imwrite('image.jpg', img)
                image_path = 'image.jpg'
                label_ = id
               
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2)
            cv2.putText(img,'Unknown Person', (x,y+400),font,2,(255, 0, 0),2)
            cv2.imwrite('image.jpg', img)
            label_ = id
            image_path = 'image.jpg'
            report_send_mail(label_, image_path)
            
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()

