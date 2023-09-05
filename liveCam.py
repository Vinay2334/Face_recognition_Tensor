import tensorflow as tf
import numpy as np
import cv2
import numpy as np
import tflearn

facedetect = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = tflearn.DNN(tflearn.layers.core.input_data(shape=[50, 50, 1]))
model.load('model/model_weights.tflearn')

def get_className(classNo):
    if classNo == 0:
        return "Vinay"
    elif classNo == 1:
        return "Tony Stark"

while True:
    success, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (50, 50))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img).reshape(-1, 50, 50, 1)
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        print(prediction)
        probabilityValue = np.amax(prediction)

        if classIndex == 0:
            cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        elif classIndex == 1:
            cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
