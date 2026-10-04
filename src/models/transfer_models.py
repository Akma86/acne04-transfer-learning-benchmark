import timm
import torch
import torch.nn as nn
import torchvision.models as models


def build_transfer_model(
    model_name: str = "mobilenet_v2",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """Builds and initializes candidate transfer learning models for acne classification.

    Supported architectures:
    - 'mobilenet_v2': Lightweight CNN designed for low-latency inference.
    - 'efficientnet_b0': Compound-scaled CNN with high parameter efficiency.
    - 'resnet50': Classic deep residual network baseline.
    - 'vit_base_patch16_224' (or 'vit'): Vision Transformer with self-attention.
    """
    model_name_lower = model_name.lower().replace("-", "_")

    if model_name_lower == "mobilenet_v2":
        weights = (
            models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        return model

    elif model_name_lower in ["efficientnet_b0", "efficientnet"]:
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        return model

    elif model_name_lower in ["resnet50", "resnet"]:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        return model

    elif "vit" in model_name_lower:
        # Utilize timm for state-of-the-art Vision Transformers
        timm_name = (
            "vit_base_patch16_224"
            if model_name_lower in ["vit", "vit_base"]
            else model_name
        )
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )
        return model

    else:
        # Fallback to general timm model creation
        try:
            return timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout_rate,
            )
        except Exception as e:
            raise ValueError(
                f"Model '{model_name}' is not recognized. Error: {e}"
            )
