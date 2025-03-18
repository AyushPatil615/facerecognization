# IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from mtcnn import MTCNN

# INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Initialize MTCNN detector
detector = MTCNN()

cap = cv.VideoCapture(1)

# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Detect faces using MTCNN
    faces = detector.detect_faces(rgb_img)
    
    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)  # Ensure no negative values

        # Extract and preprocess face
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160))  # Resize to match FaceNet input size
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Generate face embeddings and predict
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]

        # Draw bounding box and label
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)

    # Show the frame
    cv.imshow("Face Recognition:", frame)

    # Break loop on 'q' press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
