import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from utils.config import IMG_SIZE, CLASS_NAMES


def predict(model, pil_image):
    """
    Run inference on a PIL image.

    Returns:
        label      (str)   – predicted class name
        confidence (float) – probability of the top class  [0, 1]
        probs      (list)  – softmax probabilities for all classes
    """
    img = pil_image.resize(IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, 0)
    x = preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]          # shape: (num_classes,)
    idx = int(np.argmax(probs))

    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()
