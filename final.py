import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date

path = 'Images_Excel'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        print ("encode là",encode)
    return encodeList

def get_csv_filename():
    today = date.today()
    return f"Diem_danh_{today.strftime('%d-%m-%Y')}.csv"

def create_csv_file():
    filename = get_csv_filename()
    if os.path.isfile(filename):
        print(f"CSV file already exists: {filename}")
    else:
        with open(filename, 'w') as f:
            f.write('Name,Time,Valmin')
        print(f"CSV file created: {filename}")
    return filename

def Attendance(name,valmin,csv_file):
    with open(csv_file, 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%d/%m/%Y, %H:%M:%S')
        f.write(f'\n{name},{dtString},{valmin}')

encodeListKnown = findEncodings(images)
print('Endcoding OK. Loading camera...')

cap = cv2.VideoCapture(1)

csv_file = create_csv_file()
print(f"CSV file created: {csv_file}")

attending_faces = []

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for i, encodeFace in enumerate(encodesCurFrame):
        faceLoc = facesCurFrame[i]
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)
        print (encodeListKnown[matchIndex])

        if matches[matchIndex]:
             name = classNames[matchIndex].upper()
            valmin = "{}".format(round(100*(1-faceDis[matchIndex])))

            if name not in attending_faces:
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name + ' - ' + valmin +'%',(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
                Attendance(name,valmin,csv_file)
                attending_faces.append(name)

    cv2.imshow('Camera 01',img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
encodeListKnown = findEncodings(images)
def save_encodings_to_file(encodings, filename):
    with open(filename, 'w') as f:
        for encoding in encodings:
            np.savetxt(f, encoding, newline=' ')
            f.write('\n')

save_encodings_to_file(encodeListKnown, 'encodings.txt')


