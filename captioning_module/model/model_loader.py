# captioning_module/model/model_loader.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BitsAndBytesConfig
from PIL import Image
import torch
import platform

# 1. QUANT_CONFIG 정의 블록을 조건부로 변경합니다.
# platform.system()이 'Darwin' (macOS)일 경우 양자화 건너뛰기
if platform.system() != "Darwin":
    try:
        QUANT_CONFIG_GLOBAL = BitsAndBytesConfig(  # 변수명을 GLOBAL로 변경 (혼동 방지)
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
    except Exception as e:
        print(
            f"Warning: Failed to initialize BitsAndBytesConfig. Proceeding without 4-bit quantization. Error: {e}"
        )
        QUANT_CONFIG_GLOBAL = None
else:
    # macOS 환경에서는 양자화 구성을 None으로 설정하거나 사용하지 않습니다.
    QUANT_CONFIG_GLOBAL = None


class ModelLoader:
    _blip_model = None
    _blip_processor = None

    BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"

    # ⭐⭐ 수정: 전역 변수를 클래스 변수로 할당 (핵심 수정)
    QUANT_CONFIG = QUANT_CONFIG_GLOBAL

    @classmethod
    def get_blip(cls):
        if cls._blip_model is None or cls._blip_processor is None:
            cls._blip_model = BlipForConditionalGeneration.from_pretrained(
                cls.BLIP_MODEL_ID,
                quantization_config=cls.QUANT_CONFIG,
                # ⭐ 수정: low_cpu_mem_usage=False를 추가하여 meta 텐서 로딩 우회
                # device_map="auto"는 low_cpu_mem_usage=True와 연관되어 meta 로딩을 유발합니다.
                low_cpu_mem_usage=False,
            ).to(
                "cpu"
            )  # ⭐ 추가: 혹시 모를 로딩 문제에 대비해 명시적으로 CPU 이동
            cls._blip_processor = BlipProcessor.from_pretrained(cls.BLIP_MODEL_ID)
        return cls._blip_model, cls._blip_processor


# Model Loder(mac 환경에서 사용시)
class ModelLoader_mac:
    _blip_model = None
    _blip_processor = None

    BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def get_blip(cls):
        if cls._blip_model is None or cls._blip_processor is None:
            cls._blip_model = BlipForConditionalGeneration.from_pretrained(
                cls.BLIP_MODEL_ID
            ).to(cls.DEVICE)
            cls._blip_processor = BlipProcessor.from_pretrained(cls.BLIP_MODEL_ID)
        return cls._blip_model, cls._blip_processor
