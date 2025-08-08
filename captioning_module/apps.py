from django.apps import AppConfig
from captioning_module.model.model_loader import ModelLoader, ModelLoader_mac

class CaptioningModuleConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "captioning_module"
    
    def ready(self):
        ModelLoader_mac.get_clip()
        ModelLoader_mac.get_blip()
