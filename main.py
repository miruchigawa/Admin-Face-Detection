import os
import cv2
import numpy as np
import joblib
import animeface
from PIL import Image

model = joblib.load('svm_model.pkl')

def detect_is_admin_video(file):
    if not os.path.isdir('result'):
        os.mkdir('result')

    video = cv2.VideoCapture(file)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result = os.path.split(file)[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'result/{result}', fourcc, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = animeface.detect(Image.fromarray(gray))
        
        for face in faces:
            x = face.face.pos.x
            y = face.face.pos.y
            w = face.face.pos.width
            h = face.face.pos.height
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_img = Image.fromarray(gray).crop((x, y, x+w, y+h)).resize((100, 100))
            flattened_img = np.array(face_img).flatten().reshape(1, -1)
            prediction = model.predict(flattened_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            cv2.putText(frame, "Is Not Admin" if not prediction else "Is Admin", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out.write(frame)
    
    video.release()
    out.release()
    print(f'Video saved as result/{result}')


def detect_is_admin(file):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    gray = cv2.equalizeHist(gray)
    faces = animeface.detect(Image.fromarray(gray))

    for face in faces:
        x = face.face.pos.x
        y = face.face.pos.y
        w = face.face.pos.width
        h = face.face.pos.height

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_img = Image.fromarray(gray).crop((x, y, x+w, y+h)).resize((100, 100))
        flattened_img = np.array(face_img).flatten().reshape(1, -1)
        prediction = model.predict(flattened_img)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cv2.putText(image, "Is Not Admin" if not prediction else "Is Admin", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    if not os.path.isdir('result'):
        os.mkdir('result')
    
    result = os.path.split(file)[1]

    if cv2.imwrite(f'result/{result}', image):
        print(f'Result saved as result/{result}')
    else:
        print(f'Image {result} not saved')

if __name__ == "__main__":
    for file in os.listdir('test'):
        path = os.path.join('test', file)
        detect_is_admin_video(path) if file.endswith('.mp4') else detect_is_admin(path)
