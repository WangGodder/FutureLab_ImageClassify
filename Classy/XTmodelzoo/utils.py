from .resnext import pretrained_settings as resnext_settings
from .torchvision_models import pretrained_settings as torchvision_models_settings


all_settings = [
    resnext_settings,
    torchvision_models_settings,
]

model_names = []
pretrained_settings = {}
for settings in all_settings:
    for model_name, model_settings in settings.items():
        pretrained_settings[model_name] = model_settings
        model_names.append(model_name)
