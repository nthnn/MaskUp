from keras.models import load_model
from mask_detection import detect_masks

def main():
    maskup_model = load_model('../models/keras_model/maskup-model.keras')
    detect_masks(maskup_model)

if __name__ == '__main__':
    main()
