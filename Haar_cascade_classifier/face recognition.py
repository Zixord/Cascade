import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# use W:\python\educational\image_processing_mojtama_class\code\Haar_cascade_classifier>    address

def main():
    root = os.getcwd()
    imgPath = os.path.join(root,'w:\\python\\Educational\\image_processing_mojtama_class\\images\\Resistor12.jpg')
    img = cv.imread(imgPath)
    Grayimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    resistorCompoent_cascade = cv.CascadeClassifier("cascade.xml")#cascade.xml / haarcascade_frontalface_default.xml

    print("Haar cascade XML file path:", os.path.abspath("cascade.xml"))

    while True:

        
        Resistor = resistorCompoent_cascade.detectMultiScale(Grayimg, 1.1,5)

        for x,y,w,h in Resistor:
            cv.rectangle(img,(x,y),((x+w),(y+h)),(100,255,100),2)

        cv.imshow("img",img)
        
        if cv.waitKey(1) == ord('q'):
            break


    cv.destroyAllWindows()


start = input('press S to start ')
start = start.lower()
if start == 's':
    main()
