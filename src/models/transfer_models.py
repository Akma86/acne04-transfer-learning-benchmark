import numpy as np
# NumPy 2.0 compatibility shims
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

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

    Supported candidate architectures:
    - 'mobilenet_v2': Lightweight inverted residual CNN for mobile/edge diagnosis.
    - 'efficientnet_b0': Compound-scaled CNN with optimal parameter efficiency.
    - 'resnet50': Deep residual baseline network.
    - 'vit' / 'vit_b_16': Vision Transformer using self-attention across image patches.
    """
    model_name_lower = model_name.lower().replace("-", "_")

    if model_name_lower in ["mobilenet_v2", "mobilenet"]:
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

    elif model_name_lower in ["vit", "vit_b_16", "vision_transformer"]:
        # Native PyTorch Vision Transformer (ViT-B/16)
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        return model

    else:
        # Optional fallback via timm
        try:
            import timm
            return timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout_rate,
            )
        except Exception as e:
            raise ValueError(
                f"Model '{model_name}' is not recognized. Choose from "
                f"['mobilenet_v2', 'efficientnet_b0', 'resnet50', 'vit']. (Error: {e})"
            )
