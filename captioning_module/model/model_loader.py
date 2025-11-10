from optimum.intel.openvino import OVModelForVision2Seq
from transformers import AutoProcessor
from PIL import Image
import torch
import platform
import os

class ModelLoader:
    _blip_model = None
    _blip_processor = None

    # OpenVINO로 export된 모델 경로
    BLIP_MODEL_DIR = os.getenv("BLIP_OV_MODEL_DIR", "./blip_ov_int8")

    DEVICE = "CPU"

    @classmethod
    def get_blip(cls):
        """
        OpenVINO IR 기반 BLIP 모델을 로드합니다.
        최초 1회만 로딩 후 재사용합니다.
        """
        if cls._blip_model is None or cls._blip_processor is None:
            print(f"[INFO] Loading OpenVINO BLIP model from: {cls.BLIP_MODEL_DIR}")

            # OpenVINO IR 로딩 (이미 export 완료된 상태여야 함)
            cls._blip_model = OVModelForVision2Seq.from_pretrained(
                cls.BLIP_MODEL_DIR,
                device=cls.DEVICE,
                compile=True,
            )

            cls._blip_processor = AutoProcessor.from_pretrained(cls.BLIP_MODEL_DIR)
            print("[INFO] BLIP model and processor successfully loaded.")
        return cls._blip_model, cls._blip_processor


# macOS 환경 전용 로더 (MPS용)
class ModelLoader_mac:
    _blip_model = None
    _blip_processor = None
    BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def get_blip(cls):
        """
        macOS (MPS)에서 OpenVINO 미지원 시 기본 PyTorch BLIP 모델을 로드합니다.
        """
        if cls._blip_model is None or cls._blip_processor is None:
            print(f"[INFO] Loading BLIP model for macOS ({cls.DEVICE})...")
            cls._blip_model = (
                torch.compile(
                    BlipForConditionalGeneration.from_pretrained(cls.BLIP_MODEL_ID)
                ).to(cls.DEVICE)
                if hasattr(torch, "compile")
                else BlipForConditionalGeneration.from_pretrained(cls.BLIP_MODEL_ID).to(cls.DEVICE)
            )
            cls._blip_processor = AutoProcessor.from_pretrained(cls.BLIP_MODEL_ID)
        return cls._blip_model, cls._blip_processor
