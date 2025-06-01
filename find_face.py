import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model(r'face_classifier.h5')

class_labels = {0: 'Aaron_Eckhart', 1: 'other'} 

# Initialize Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_face(face_image):
    img = cv2.resize(face_image, (200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0 
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    return class_labels[predicted_class], confidence

def main():
    cap = cv2.VideoCapture(0)  
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            
            label, confidence = predict_face(face_image)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1)
        if key == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
