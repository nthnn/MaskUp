import cv2
import numpy as np
import os

from dictionaries import label_dict, color_dict
from face_detection import detect_faces
from image_processing import preprocess_image
from keras.models import load_model
from keras.utils import load_img, img_to_array
from visualization import draw_bounding_box

def predict_mask(face_img, model):
    face_img = preprocess_image(face_img)
    cv2.imwrite('temp.jpg', face_img)

    test_image = load_img('temp.jpg', target_size=(150, 150, 3))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    pred = model.predict(test_image)[0][0] == 0
    os.remove('temp.jpg')

    return pred

def detect_masks(maskup_model):
    video = cv2.VideoCapture(0)

    while video.isOpened():
        _, img = video.read()
        faces = detect_faces(img)

        for (x, y, w, h) in faces:
            face_img = img[y: y + h, x: x + w]
            pred = predict_mask(face_img, maskup_model)

            draw_bounding_box(img, x, y, w, h, pred)
            cv2.putText(img, label_dict[pred], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('MaskUp', img)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
