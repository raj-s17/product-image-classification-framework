"""
Central configuration — import from here, never redefine elsewhere.
"""

IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "Accessories",
    "Apparel",
    "Footwear",
    "Free Items",
    "Personal Care",
]

CLASS_ICONS = {
    "Accessories": "👜",
    "Apparel": "👕",
    "Footwear": "👟",
    "Free Items": "🎁",
    "Personal Care": "🧴",
}

MODEL_PATH = "models/resnet50_final.keras"

# Grad-CAM target layer (last conv block of ResNet50)
GRADCAM_LAYER = "conv5_block3_out"
