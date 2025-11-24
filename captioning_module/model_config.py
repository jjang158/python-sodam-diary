import os

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
# BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BLIP_MODEL_DIR = os.path.join(
    FILE_DIR, 
    # "blip_openvino_base",
    "blip_openvino",
)
BLIP_MODEL_PATH = os.path.join(
    BLIP_MODEL_DIR, "blip_caption.xml"
)
