#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ricardo
@license: (C) Copyright 2018-2019 @yang.com Corporation Limited.
@contact: 659706575@qq.com
@software: made@Yang
@file: face_cv.py
@time: 2019/1/26 0026 14:33
@desc:
'''
def face_dect_img():

    import cv2

    img = cv2.imread('faces.jpg')

    pattern = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = pattern.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5,minSize=(5,5))

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imwrite('ret_img2.jpg',img)


def face_detect_video():
    import cv2
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

    cap = cv2.VideoCapture(0)
    while True:
        ret, img  = cap.read()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = img[y:y+h,x:x+w]

            eyes =  eye_cascade.detectMultiScale(roi_gray,1.3,5)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xff ==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
# face_detect_video()
def face_draw_video():
    import cv2
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    url = 'http://yang:19880927@192.168.0.140:8989/video'
    cap = cv2.VideoCapture(url)
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)  # 前置摄像头 左右改变
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            text = 'width:{}  height:{}'.format( w, h)
            frame = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX,0.7, (255, 255, 0),2)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

face_draw_video()
# from hyperlpr import  pipline as  pp
# import cv2
# # 自行修改文件名
# image = cv2.imread("faces.jpg")
# image,res  = pp.SimpleRecognizePlate(image)
# print(res)