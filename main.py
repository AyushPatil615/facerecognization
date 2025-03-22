from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import base64
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from mtcnn import MTCNN

app = Flask(__name__)

# Load models and encoders
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))
detector = MTCNN()

def recognize_face(img):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_img)
    
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)
        face_img = rgb_img[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)

        ypred = facenet.embeddings(face_img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]
        return final_name
    
    return "No face detected"

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({"status": "error", "message": "No image provided"}), 400
        
        # Decode base64 image
        img_data = base64.b64decode(data)
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv.imdecode(np_img, cv.IMREAD_COLOR)

        result = recognize_face(img)
        return jsonify({"status": "success", "name": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)

