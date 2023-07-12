import cv2

from dictionaries import color_dict

def draw_bounding_box(img, x, y, w, h, pred):
    cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[pred], 2)
    cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[pred], -1)
