from io import BytesIO
import numpy as np
from PIL import Image
import openvino as ov
from transformers import BlipProcessor
from typing import Optional, Any
import time
from .model_config import BLIP_MODEL_ID, BLIP_MODEL_PATH

class ImageCaptioner:

    MIN_TOKEN = 10
    MAX_TOKEN = 20

    _this = None

    @classmethod
    def get_image_captioner(cls):
        """
        싱글톤 인스턴스 반환
        """
        if cls._this is None:
            cls._this = cls()   # 최초 1회만 생성
        return cls._this

    def __init__(
        self,
        ov_model_path: str = BLIP_MODEL_PATH,
        device: str = "AUTO",
    ):
        """
        OpenVINO 기반 BLIP 이미지 캡셔너 초기화
        """
        if ImageCaptioner._this is not None:
            return  

        self.processor = BlipProcessor.from_pretrained(
            BLIP_MODEL_ID
        )

        image_processor = self.processor.image_processor
        size = image_processor.size
        # size가 dict인 경우
        if isinstance(size, dict):
            self.image_height = int(size.get("height"))
            self.image_width = int(size.get("width"))
        else:
            # 그냥 정수 하나로 오는 경우
            self.image_height = int(size)
            self.image_width = int(size)
        self.image_mean = np.array(image_processor.image_mean, dtype=np.float32)
        self.image_std = np.array(image_processor.image_std, dtype=np.float32)

        core = ov.Core()
        ov_model = core.read_model(ov_model_path)
        self.compiled_model = core.compile_model(ov_model, device)
        self.output = self.compiled_model.output(0)

        self.tokenizer = self.processor.tokenizer
        self.bos_token_id = (
            self.tokenizer.bos_token_id
            or self.tokenizer.cls_token_id
            or self.tokenizer.pad_token_id
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        print(f"BLIP_MODEL_PATH: {ov_model_path}")

        print("[Singleton] OpenVINO BLIP Captioner Loaded")

    # ----------------------------------------------------
    # 이미지 분석
    # ----------------------------------------------------
    def get_blip_analyze(self, image_bytes: bytes) -> str:
        t0 = time.perf_counter()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        print("[INFO] Generating BLIP caption...")
        caption = self._generate_caption(image)
        t1 = time.perf_counter()
        print(f"[PROFILE] Total caption time: {(t1 - t0):.3f} sec")
        print(f"[INFO] Generated Caption: success")
        return caption

    # ----------------------------------------------------
    # 내부 기능
    # ----------------------------------------------------
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # 1) 리사이즈 (BLIP는 보통 384 기준)
        image = image.resize((self.image_width, self.image_height))

        # 2) numpy 배열로 변환 (H, W, C)
        arr = np.array(image).astype(np.float32) / 255.0

        # 3) 정규화 (mean/std는 (3,)이라 브로드캐스트 됨)
        arr = (arr - self.image_mean) / self.image_std

        # 4) CHW로 transpose
        arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)

        # 5) 배치 차원 추가 → (1, 3, H, W)
        return np.expand_dims(arr, axis=0)


    def _generate_caption(
        self, image: Image.Image, max_new_tokens: int = MAX_TOKEN, min_new_tokens: int = MIN_TOKEN
    ) -> str:
        t0 = time.perf_counter()
        pixel_values = self._preprocess(image)
        t1 = time.perf_counter()
        print(f"[PROFILE] Preprocess time: {(t1 - t0):.3f} sec")

        input_ids = np.array([[self.bos_token_id]], dtype=np.int64)

        for step in range(max_new_tokens):
            t_loop0 = time.perf_counter()
            outputs = self.compiled_model({0: pixel_values, 1: input_ids})
            logits = outputs[self.output]
            t_loop1 = time.perf_counter()
            print(f"[PROFILE] Step {step+1} infer: {(t_loop1 - t_loop0):.3f} sec")

            next_token_logits = logits[:, -1, :]
            next_token_id = int(next_token_logits.argmax(axis=-1)[0])

            input_ids = np.concatenate(
                [input_ids, np.array([[next_token_id]], dtype=np.int64)],
                axis=1,
            )

            if (
                self.eos_token_id is not None
                and step >= min_new_tokens
                and next_token_id == self.eos_token_id
            ):
                break

        caption = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        print(f"[DEBUG] Raw generated caption: {caption}")
        return caption.strip()