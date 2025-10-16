from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPModel, CLIPProcessor, BitsAndBytesConfig
from PIL import Image
import torch
import platform

# 1. QUANT_CONFIG 정의 블록을 조건부로 변경합니다.
# platform.system()이 'Darwin' (macOS)일 경우 양자화 건너뛰기
if platform.system() != "Darwin":
    # Python이 bitsandbytes 패키지를 찾을 때 발생할 수 있는 오류를 방지하기 위해 try/except로 감쌉니다.
    try:
        QUANT_CONFIG = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
    except Exception as e:
        print(
            f"Warning: Failed to initialize BitsAndBytesConfig. Proceeding without 4-bit quantization. Error: {e}"
        )
        QUANT_CONFIG = None
else:
    # macOS 환경에서는 양자화 구성을 None으로 설정하거나 사용하지 않습니다.
    QUANT_CONFIG = None


class ModelLoader:
    _clip_model = None
    _clip_processor = None
    _blip_model = None
    _blip_processor = None

    BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

    # QUANT_CONFIG = BitsAndBytesConfig(
    #     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    # )

    @classmethod
    def get_blip(cls):
        if cls._blip_model is None or cls._blip_processor is None:
            cls._blip_model = BlipForConditionalGeneration.from_pretrained(
                cls.BLIP_MODEL_ID,
                quantization_config=cls.QUANT_CONFIG,
                device_map="auto",
            )
            cls._blip_processor = BlipProcessor.from_pretrained(cls.BLIP_MODEL_ID)

        return cls._blip_model, cls._blip_processor

    @classmethod
    def get_clip(cls):
        if cls._clip_model is None or cls._clip_processor is None:
            cls._clip_model = CLIPModel.from_pretrained(
                cls.CLIP_MODEL_ID,
                quantization_config=cls.QUANT_CONFIG,
                device_map="auto",
            )
            cls._clip_processor = CLIPProcessor.from_pretrained(cls.CLIP_MODEL_ID)

        return cls._clip_model, cls._clip_processor


# Model Loder(mac 환경에서 사용시)
class ModelLoader_mac:
    _clip_model = None
    _clip_processor = None
    _blip_model = None
    _blip_processor = None

    BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def get_blip(cls):
        if cls._blip_model is None or cls._blip_processor is None:
            cls._blip_model = BlipForConditionalGeneration.from_pretrained(
                cls.BLIP_MODEL_ID
            ).to(cls.DEVICE)
            cls._blip_processor = BlipProcessor.from_pretrained(cls.BLIP_MODEL_ID)
        return cls._blip_model, cls._blip_processor

    @classmethod
    def get_clip(cls):
        if cls._clip_model is None or cls._clip_processor is None:
            cls._clip_model = CLIPModel.from_pretrained(cls.CLIP_MODEL_ID).to(
                cls.DEVICE
            )
            cls._clip_processor = CLIPProcessor.from_pretrained(cls.CLIP_MODEL_ID)
        return cls._clip_model, cls._clip_processor
