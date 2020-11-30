import os,sys
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/Users/ntt/.pyenv/versions/3.6.9/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
counter=0
mabatakilist=[]
average_eye=0 #目の大きさの平均
average_bit=0 #目の大きさの平均をとるタイミング
testbit=0
sl=5 #データの極小値を見つける範囲(order)
listsize=0
MaxFlame=1000
timelist=[]


def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

def eye_marker(face_mat, position):
    for i, ((x, y)) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)  

def mouth_marker(face_mat, position):
    for i, ((x, y)) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)  
        
while True:
    tick = cv2.getTickCount()

    ret, rgb = cap.read()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
    

    if len(faces) == 1:
        e1 = cv2.getTickCount()
        x, y, w, h = faces[0, :]

        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_gray = gray[y :(y + h), x :(x + w)]
        scale = 480 / h
        face_gray_resized = cv2.resize(face_gray, dsize=None, fx=scale, fy=scale)

        face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
        face_parts = face_parts_detector(face_gray_resized, face)
        face_parts = face_utils.shape_to_np(face_parts)

        left_eye = face_parts[42:48]
        mouth = face_parts[60:65]
        eye_marker(face_gray_resized, left_eye)
        mouth_marker(rgb,mouth)

        left_eye_ear = calc_ear(left_eye)
        cv2.putText(rgb, "LEFT eye EAR:{} ".format(left_eye_ear), 
            (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        right_eye = face_parts[36:42]
        eye_marker(face_gray_resized, right_eye)

        right_eye_ear = calc_ear(right_eye)
        cv2.putText(rgb, "RIGHT eye EAR:{} ".format(round(right_eye_ear, 3)), 
            (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        mabatakilist.append(left_eye_ear+right_eye_ear)
        #print(len(mabatakilist))

        listsize=listsize+1

        #目の開き具合の平均値測定
        if listsize==50 and average_bit==0:
            average_eye=sum(mabatakilist)/len(mabatakilist)
            average_bit=1
            print(average_eye)
            #閾値設定
            mabataki_close=average_eye*7/10 #瞬きと判断する閾値
            mabataki_open=average_eye*2/3
            frame_ave=sum(timelist)/len(timelist)
            print("frame_ave",frame_ave)


        if listsize==100 and testbit==0:
            print("save")
            testbit=1

        #瞬きを検知する
        if listsize % sl == 0 and average_bit==1:
            start=listsize-sl+1
            print("hani=",start,",",listsize)#極小値を見つける範囲
            #sgf = signal.savgol_filter(mabatakilist[start:listsize], sl, 2, deriv=0)
            
            x=np.array(mabatakilist[start:listsize])
            #x=np.array(sgf)
            minList= signal.argrelmin(x,order=sl)
            #極小値がmabataki_closeの値以下だったら瞬きと検知する
            print("minList=",minList[0])
            for min in minList[0]:
                print("min=",start+min)
                print("value=",mabatakilist[start+min])
                plt.plot(start+min,mabatakilist[start+min], "bo")
                if mabatakilist[start+min]<mabataki_close:
                    print("close value=",mabatakilist[start+min])
                    counter=counter+1
            minList=np.delete(minList, 0, 0)
            #print("del minList=",minList)


        if listsize > MaxFlame:
            print("delete")
            del mabatakilist[1:10]
            listsize=listsize-10
            #print("a="+str(sum(mabatakilist[1:10])/10))
            #print("b="+str(sum(mabatakilist[41:50])/10))


        
        
        
        cv2.imshow('frame_resize', face_gray_resized)
        e2 = cv2.getTickCount()
        if listsize<51:
            timelist.append((e2 - e1)/ cv2.getTickFrequency())
    cv2.putText(rgb,"counter="+str(counter), (10,180), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, 1)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(rgb, "FPS:{} ".format(int(fps)), 
        (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    

    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) == 27:  
        plt.plot(mabatakilist)  
        #sgf = signal.savgol_filter(mabatakilist, sl, 2, deriv=0)
        #plt.plot(sgf) 
        t = np.arange(0, listsize, 1)   
        y1=average_eye+t*0
        y2=mabataki_close+t*0
        plt.plot(t, y1)
        plt.plot(t, y2)
        plt.savefig("image.png")
        print("mabataki=",counter)
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()