# Lazy imports to avoid requiring open_clip at import time
from .mlp_classifier import MLPClassifier
from .threshold_classifier import ThresholdClassifier


def get_clip_encoder():
    from .clip_encoder import CLIPEncoder
    return CLIPEncoder
