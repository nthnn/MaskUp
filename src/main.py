import cv2
import numpy as np

from keras.models import load_model

from face_detector import detect_faces
from mask_classifier import classify_mask

results = {
    0: 'Without mask',
    1: 'Mask'
}

color_dict = {
    0: (0, 0, 255),
    1: (0, 255, 0)
}

rect_size = 4

cap = cv2.VideoCapture(0)
model = load_model('../models/keras_model/maskup-model.keras')
haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()
    im = cv2.flip(im, 1, 1)

    faces = detect_faces(im, haarcascade, rect_size)

    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y: y + h, x: x + w]
        label = classify_mask(model, face_img)

        cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('MaskUp', im)
    key = cv2.waitKey(10)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
