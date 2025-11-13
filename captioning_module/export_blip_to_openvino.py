import os
from pathlib import Path

import torch
import openvino as ov
from transformers import BlipForConditionalGeneration, BlipProcessor
from model_config import BLIP_MODEL_ID, BLIP_MODEL_DIR

OUTPUT_DIR = Path(BLIP_MODEL_DIR)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. HuggingFace 모델/프로세서 로드
    print("Loading BLIP model & processor from Hugging Face...")
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model.eval()

    # 2. 예제 입력(example_input) 준비
    dummy_pixel_values = torch.randn(1, 3, 384, 384)
    bos_token_id = processor.tokenizer.bos_token_id
    if bos_token_id is None:
        # 혹시 bos_token이 없는 토크나이저면 pad/cls 중 하나 사용
        bos_token_id = (
            processor.tokenizer.cls_token_id
            or processor.tokenizer.pad_token_id
        )
    dummy_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long)
    print("Converting PyTorch BLIP model to OpenVINO IR...")

    # 3. PyTorch -> OpenVINO Model 변환
    ov_model = ov.convert_model(
        model,
        example_input=(dummy_pixel_values, dummy_input_ids),
    )

    # 4. IR 모델 저장 (기본적으로 FP16 압축)
    xml_path = OUTPUT_DIR / "blip_caption.xml"
    ov.save_model(ov_model, xml_path) 

    print(f"OpenVINO IR saved to: {xml_path.resolve()}")
    print("변환 완료!")

if __name__ == "__main__":
    main()
