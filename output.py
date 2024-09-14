import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('best_model/best_model.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_image(img, target_size=(48, 48)):
    img_resized = cv2.resize(img, target_size)
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0
    return img_array


# Function to draw bounding box and predict emotion
def display_result_with_bbox_color(img_path, model):
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    img = cv2.imread(img_path)
    orig_img = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray_img[y:y + h, x:x + w]

        processed_face = preprocess_image(face_roi)

        prediction = model.predict(processed_face)
        predicted_class = class_labels[np.argmax(prediction)]

        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(orig_img, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    cv2.imwrite('output_image_with_bbox.jpg', img_rgb)


# Example usage
img_path = 'surprise.jpg'
display_result_with_bbox_color(img_path, model)
