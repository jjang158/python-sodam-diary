from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPModel, CLIPProcessor, BitsAndBytesConfig
from PIL import Image
import torch


class ModelLoader:
    _clip_model = None
    _clip_processor = None
    _blip_model = None
    _blip_processor = None
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    @classmethod
    def get_blip(cls):
        if cls._blip_model is None or cls._blip_processor is None:
            blip_model_id = "Salesforce/blip-image-captioning-base"

            cls._blip_model = BlipForConditionalGeneration.from_pretrained(
                blip_model_id,
                quantization_config=cls.quant_config,
                device_map="auto"
            )
            cls._blip_processor = BlipProcessor.from_pretrained(blip_model_id)

        return cls._blip_model, cls._blip_processor
    
    @classmethod
    def get_clip(cls):
        if cls._clip_model is None or cls._clip_processor is None:
            clip_model_id = "openai/clip-vit-base-patch32"

            cls._clip_model = CLIPModel.from_pretrained(
                clip_model_id,
                quantization_config=cls.quant_config,
                device_map="auto"
            )
            cls._clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

        return cls._clip_model, cls._clip_processor
    