from concurrent.futures import ThreadPoolExecutor
from .model_loader import ModelLoader, ModelLoader_mac
from PIL import Image, UnidentifiedImageError
import torch
import logging
from io import BytesIO  # BytesIO 모듈을 가져옵니다.

logger = logging.getLogger(__name__)
# macOS 환경에서는 'mps'를 사용하거나 'cpu'를 사용합니다.
# AWS 환경에 맞게 CPU로 설정
DEVICE = torch.device('cpu')

MOODS = [
    "평화로움", "따뜻함", "쓸쓸함", "기쁨", "설렘", "슬픔", "여유로움", "고독함", "아픔", "행복",
    "차분함", "긴장감", "만족감", "행복감", "분노", "불안", "낭만", "공허함", "기대", "밝음",
    "희망", "후련함", "흥분", "피곤함", "잔잔함", "신비로움", "몽환적", "우울함", "즐거움", "편안함", "진중함"
]

def analyze_image(image_data):
    """
    바이트 형태의 이미지 데이터를 받아 비동기적으로 BLIP과 CLIP 분석을 수행합니다.
    """
    # 비동기방식
    with ThreadPoolExecutor() as executor:
        # BytesIO를 사용하여 메모리 내에서 파일처럼 읽을 수 있는 객체를 생성합니다.
        # 이렇게 하면 각 분석 함수가 파일을 다시 읽을 필요 없이 동일한 메모리 데이터를 사용합니다.
        future_blip = executor.submit(get_blip_analyze, image_data)
        future_clip = executor.submit(get_clip_analyze, image_data)

        caption = future_blip.result()
        moods = future_clip.result()
    
    return {
        "file_description": caption,
        "file_moods": moods
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

        out = blip_model.generate(**inputs,
            max_new_tokens=50,
            min_length=30,
            early_stopping=True)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
    except UnidentifiedImageError:
        logger.warning("유효하지 않은 이미지 파일입니다.")
        caption = None
    except Exception as e:
        logger.exception("BLIP 분석 실패: %s", e)
        caption = None
    
    return caption


# CLIP 모델을 통한 사진 분석(분위기)
def get_clip_analyze(image_data):
    """
    바이트 형태의 이미지 데이터를 CLIP 모델로 분석하여 분위기를 추출합니다.
    """
    result = []
    try:
        # 1. 모델 로드 (메모리에서 가져옴)
        # AWS 환경에 맞게 ModelLoader 사용
        clip_model, clip_processor = ModelLoader.get_clip()
        
        # 2. 이미지 데이터 준비
        # BytesIO를 사용하여 바이트 데이터를 PIL Image로 엽니다.
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        inputs = clip_processor(text=MOODS, images=image, return_tensors="pt", padding=True)
        # 모든 입력 텐서를 명시적으로 DEVICE로 이동시킵니다.
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)

        
        # CLIP 추론
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
        # 분위기 추출 (가장 근접한 3개)
        top_k = torch.topk(probs, k=3)
        top_indices = top_k.indices.squeeze().tolist()
        top_scores = top_k.values.squeeze().tolist()

        # 리스트가 1개일 경우
        if isinstance(top_indices, int):
            top_indices = [top_indices]
            top_scores = [top_scores]

        # 결과 출력
        print("예측된 분위기:")
        
        for idx, score in zip(top_indices, top_scores):
            obj = {}
            obj['label'] = MOODS[idx]
            obj['score'] = round(score*100, 2)
            result.append(obj)
    except UnidentifiedImageError:
        logger.warning("유효하지 않은 이미지 파일입니다.")
    except Exception as e:
        logger.exception("CLIP 분석 실패: %s", e)
    
    return result
