import animeface
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import joblib

def load_dataset(path):
    images = []
    labels = []
    
    for label in os.listdir(path):
        label_dir = os.path.join(path, label)
        
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path)
            
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
                flattened_img = np.array(face_img).flatten()
             
                images.append(flattened_img)
                labels.append(1 if label == "konata" else 0)

    return np.array(images), np.array(labels)

images, labels = load_dataset("dataset")

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Train Accuracy: {train_accuracy}")

y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy}")

joblib.dump(model, 'svm_model.pkl')
print('Model saved as svm_model.pkl')
