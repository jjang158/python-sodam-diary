from concurrent.futures import ThreadPoolExecutor
from .model_loader import ModelLoader, ModelLoader_mac
from PIL import Image, UnidentifiedImageError
import torch
import logging
from io import BytesIO  # BytesIO 모듈을 가져옵니다.

logger = logging.getLogger(__name__)
# macOS 환경에서는 'mps'를 사용하거나 'cpu'를 사용합니다.
# AWS 환경에 맞게 CPU로 설정
DEVICE = torch.device("cpu")


def analyze_image(image_data):
    """
    바이트 형태의 이미지 데이터를 받아 비동기적으로 BLIP을 통해 분석을 수행합니다.
    """
    # 이미지 묘사
    caption = get_blip_analyze(image_data)
    return {
        "file_description": caption,
    }


# BLIP 모델을 통한 사진 분석(사진묘사)
def get_blip_analyze(image_data):
    """
    바이트 형태의 이미지 데이터를 BLIP 모델로 분석하여 캡션을 생성합니다.
    """
    try:
        # 1. 모델 로드 (메모리에서 가져옴)
        # AWS 환경에 맞게 ModelLoader 사용
        blip_model, blip_processor = ModelLoader.get_blip()
        print("blip model loaded")

        # 2. 이미지 데이터 준비
        # BytesIO를 사용하여 바이트 데이터를 PIL Image로 엽니다.
        image = Image.open(BytesIO(image_data)).convert("RGB")
        print("image loaded.")

        inputs = blip_processor(images=image, return_tensors="pt")
        print("inputs loaded.")

        # 모든 입력 텐서를 명시적으로 DEVICE로 이동시킵니다.
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)

        out = blip_model.generate(
            **inputs, max_new_tokens=50, min_length=30, early_stopping=True
        )
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
    except UnidentifiedImageError:
        logger.warning("유효하지 않은 이미지 파일입니다.")
        caption = None
    except Exception as e:
        logger.exception("BLIP 분석 실패: %s", e)
        caption = None

    return caption
