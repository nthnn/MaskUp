import cv2
import numpy as np

def classify_mask(model, face_img):
    rerect_sized = cv2.resize(face_img, (150, 150))
    normalized = rerect_sized / 255.0

    reshaped = np.reshape(normalized, (1, 150, 150, 3))
    reshaped = np.vstack([reshaped])

    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    return label
