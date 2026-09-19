"""
DiffMark Extensions Package
10 Original Research Directions for DiffMark
"""

from .extension1_modern_deepfakes import ModernDeepfakeNoiseLayers
from .extension2_efficient_inference import EfficientDiffMarkPipeline
from .extension3_video_watermarking import VideoWatermarkPipeline
from .extension4_adaptive_guidance import AdaptiveDiffusionSampler
from .extension5_privacy_preserving import PrivacyPreservingDiffMarkTrainer
from .extension6_open_vocabulary import OpenVocabularyWatermarker
from .extension7_cross_modal import CrossModalWatermarker
from .extension8_self_supervised import SelfSupervisedDiffMarkTrainer
from .extension9_benchmark import BenchmarkSuite
from .extension10_removal_defense import RobustWatermarker

__all__ = [
    "ModernDeepfakeNoiseLayers",
    "EfficientDiffMarkPipeline",
    "VideoWatermarkPipeline",
    "AdaptiveDiffusionSampler",
    "PrivacyPreservingDiffMarkTrainer",
    "OpenVocabularyWatermarker",
    "CrossModalWatermarker",
    "SelfSupervisedDiffMarkTrainer",
    "BenchmarkSuite",
    "RobustWatermarker",
]
