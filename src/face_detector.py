import cv2

def detect_faces(image, haarcascade, rect_size):
    rerect_size = cv2.resize(image, (image.shape[1] // rect_size, image.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    return [(x, y, w, h) for (x, y, w, h) in faces]
