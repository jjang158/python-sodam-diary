from django.apps import AppConfig
from captioning_module.model.model_loader import ModelLoader, ModelLoader_mac
# if torch.backends.mps.is_available():
#     from your_module import ModelLoader_mac as ModelLoader
# else:
#     from your_module import ModelLoader

class CaptioningModuleConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "captioning_module"
    
    def ready(self):
        ModelLoader.get_clip()
        ModelLoader.get_blip()
