import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from utils.config import IMG_SIZE, GRADCAM_LAYER

def generate_gradcam(model, pil_image):

    img = pil_image.resize(IMG_SIZE)
    img = np.array(img).astype("float32")
    img = np.expand_dims(img, 0)
    processed = preprocess_input(img)

    # ResNet50 backbone
    base_model = model.layers[1]

    with tf.GradientTape() as tape:

        conv_outputs = base_model(
            processed,
            training=False
        )

        tape.watch(conv_outputs)

        x = conv_outputs

        # Pass through classifier head
        for layer in model.layers[2:]:
            x = layer(
                x,
                training=False
            )

        predictions = x

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    heatmap = tf.reduce_sum(
        conv_outputs[0] * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    original = np.array(pil_image.resize(IMG_SIZE))

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay