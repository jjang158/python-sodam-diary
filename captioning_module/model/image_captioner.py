from concurrent.futures import ThreadPoolExecutor
from model_loader import ModelLoader, ModelLoader_mac
from PIL import Image, UnidentifiedImageError
import torch
import logging

logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MOODS = [
    "평화로움", "따뜻함", "쓸쓸함", "기쁨", "설렘", "슬픔", "여유로움", "고독함", "아픔", "행복",
    "차분함", "긴장감", "만족감", "행복감", "분노", "불안", "낭만", "공허함", "기대", "밝음",
    "희망", "후련함", "흥분", "피곤함", "잔잔함", "신비로움", "몽환적", "우울함", "즐거움", "편안함", "진중함"
]

def analyze_image(file) :
    # 동기방식
    # result = {
    #     "file_description": get_blip_analyze(file),
    #     "file_moods": get_clip_analyze(file)
    # }
    # return result
    
    # 비동기방식
    with ThreadPoolExecutor() as executor:
        future_blip = executor.submit(get_blip_analyze, file)
        future_clip = executor.submit(get_clip_analyze, file)

        caption = future_blip.result()
        moods = future_clip.result()
    return {
        "file_description": caption,
        "file_moods": moods
    }


# BLIP 모델을 통한 사진 분석(사진묘사)
def get_blip_analyze(file):
    try:
        # 1. 모델 로드 (메모리에서 가져옴)
        blip_model, blip_processor = ModelLoader_mac.get_blip()
        
        # 2. 이미지 및 텍스트 준비
        image = Image.open(file).convert("RGB")
        
        inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE)

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
def get_clip_analyze(file):
    result = []
    try:
        # 1. 모델 로드 (메모리에서 가져옴)
        clip_model, clip_processor = ModelLoader_mac.get_clip()
        
        # 2. 이미지 및 텍스트 준비
        image = Image.open(file).convert("RGB")
        
        inputs = clip_processor(text=MOODS, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}
        
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
