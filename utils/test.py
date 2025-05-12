from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model(r'C:\Users\dawid\Desktop\FaceRecognitionWMA4\face_classifier.h5')

class_labels = {0: 'Aaron_Eckhart', 1: 'other'} 

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    print(f"Predicted: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")

predict_image(r'C:\Users\dawid\Desktop\FaceRecognitionWMA4\organized_faces\Aaron_Eckhart\000011.jpg')
predict_image(r'C:\Users\dawid\Desktop\FaceRecognitionWMA4\organized_faces\other\Aaron_Patterson_0001.jpg')
