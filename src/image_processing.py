import cv2

def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    return img
