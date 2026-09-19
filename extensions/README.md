# DiffMark Extensions: 10 Original Research Directions

This package contains 10 novel extensions to the original DiffMark watermarking system, each representing a potential research direction.

## Quick Start

```python
from extensions import ModernDeepfakeNoiseLayers
from extensions import EfficientDiffMarkPipeline
from extensions import VideoWatermarkPipeline
# ... and more!
```

## Extensions Overview

### 1. Modern Deepfake Noise Layers (`extension1_modern_deepfakes.py`)
- Adds support for 10+ modern deepfake methods
- Includes InsightFace, FaceSwapper/Roop, SDXL-based methods
- Video deepfakes with First Order Motion Model and LivePortrait
- More realistic and challenging attacks

**Usage**:
```python
from extensions import ModernDeepfakeNoiseLayers

noise_layers = ModernDeepfakeNoiseLayers()
watermarked = noise_layers.apply_specific_noise(img, "insightface_swap")
```

---

### 2. Efficient Real-Time Inference (`extension2_efficient_inference.py`)
- Distilled models for speed
- INT8/FP16 quantization
- TensorRT/ONNX Runtime support
- Skip DDIM steps for low latency
- Benchmarking tools

**Usage**:
```python
from extensions import EfficientDiffMarkPipeline

pipeline = EfficientDiffMarkPipeline(quantization="fp16")
watermarked, bits = pipeline.embed(img, "MyMessage")
pipeline.benchmark(img, num_runs=100)  # Measure FPS
```

---

### 3. Video Watermarking Extension (`extension3_video_watermarking.py`)
- Temporal consistency across frames
- Handles video compression
- Inter-frame watermark synchronization
- FFmpeg integration

**Usage**:
```python
from extensions import VideoWatermarkPipeline

pipeline = VideoWatermarkPipeline()
pipeline.embed_temporal_watermark("video.mp4", "watermarked.mp4", "Secret")
is_valid, msg, conf = pipeline.verify_temporal_watermark("watermarked.mp4")
```

---

### 4. Adaptive Message-Guided Sampling (`extension4_adaptive_guidance.py`)
- Learns optimal guidance scales
- Adaptive based on image content
- More guidance for faces, less for smooth backgrounds
- Trainable guidance predictor

**Usage**:
```python
from extensions import AdaptiveDiffusionSampler

sampler = AdaptiveDiffusionSampler()
opt_scale = sampler.predict_guidance_scale(img)  # Per-image scale
```

---

### 5. Privacy-Preserving Training (`extension5_privacy_preserving.py`)
- Federated training simulation
- Differential privacy (DP)
- Secure multi-party computation (SMC)
- Gradient noise injection

**Usage**:
```python
from extensions import PrivacyPreservingDiffMarkTrainer

trainer = PrivacyPreservingDiffMarkTrainer()
trainer.train_with_privacy(method="federated", use_dp=True)
```

---

### 6. Open-Vocabulary Messages (`extension6_open_vocabulary.py`)
- Variable-length messages
- JSON payload support
- Reed-Solomon error correction
- Arbitrary text/structured data

**Usage**:
```python
from extensions import OpenVocabularyWatermarker

watermarker = OpenVocabularyWatermarker(use_ecc=True)

# Embed JSON
json_payload = {"author": "John", "timestamp": 12345}
bits = watermarker.embed_json(json_payload)

# Extract
result = watermarker.extract_message(bits)
print(result["json"])  # Recovered JSON
```

---

### 7. Cross-Modal Watermarking (`extension7_cross_modal.py`)
- Pixel-level watermarking
- Audio watermarking (for video)
- Metadata (EXIF, PNG chunks)
- Multi-modal verification

**Usage**:
```python
from extensions import CrossModalWatermarker

watermarker = CrossModalWatermarker()
watermarker.embed_all("img.png", "wm.png", "Secret", metadata={"v": 1})
verification = watermarker.verify_all("wm.png")
```

---

### 8. Self-Supervised Pre-Training (`extension8_self_supervised.py`)
- Contrastive learning
- Pre-train on CelebA-HQ + FFHQ + VGGFace2
- Improved data efficiency
- Fine-tuning for watermarking

**Usage**:
```python
from extensions import SelfSupervisedDiffMarkTrainer

trainer = SelfSupervisedDiffMarkTrainer()
trainer.pre_train(num_steps=1000)
fine_tuned = trainer.fine_tune_watermarking(my_watermark_model)
```

---

### 9. Benchmark Suite (`extension9_benchmark.py`)
- 20+ attack methods
- Standard evaluation protocol
- Leaderboard generation
- PSNR, SSIM, bit accuracy metrics

**Usage**:
```python
from extensions import BenchmarkSuite

suite = BenchmarkSuite()
results = suite.benchmark(watermarker, test_images)
suite.generate_report()
suite.show_leaderboard()
```

---

### 10. Defense Against Watermark Removal (`extension10_removal_defense.py`)
- Adversarial training against removal networks
- Robust invisible watermarking
- Active detection of removal attempts

**Usage**:
```python
from extensions import RobustWatermarker

watermarker = RobustWatermarker()
watermarker.trainer.train(num_steps=500)
wm_img, bits = watermarker.embed_robust(img, "Secret")
is_removed, score = watermarker.detect_removal(attacked_img)
```

---

## Running Demos

Each extension has its own demo function:

```python
# From the watermark directory
python -c "from extensions.extension1_modern_deepfakes import demo_modern_deepfakes; demo_modern_deepfakes()"
python -c "from extensions.extension2_efficient_inference import demo_efficient_inference; demo_efficient_inference()"
# ... and so on!
```

Or run all demos:
```python
from extensions import (
    demo_modern_deepfakes,
    demo_efficient_inference,
    demo_video_watermarking,
    demo_adaptive_guidance,
    demo_privacy_preserving,
    demo_open_vocabulary,
    demo_cross_modal,
    demo_self_supervised,
    demo_benchmark,
    demo_removal_defense
)

demo_modern_deepfakes()
demo_efficient_inference()
# ... etc!
```

---

## Research Opportunities

Each extension represents a publishable research direction:

1. **Modern Deepfake Noise Layers** - Compare against original SimSwap/StarGAN
2. **Efficient Inference** - Speed/quality tradeoff studies
3. **Video Watermarking** - Temporal robustness evaluation
4. **Adaptive Guidance** - Learned vs fixed guidance scales
5. **Privacy-Preserving** - DP-SGD with different epsilon values
6. **Open-Vocabulary** - ECC vs no ECC for different payload sizes
7. **Cross-Modal** - Combined verification vs single modality
8. **Self-Supervised** - Pre-training data efficiency
9. **Benchmark Suite** - Community challenge
10. **Removal Defense** - Adversarial training ablation studies

---

## Citation

If you use these extensions, please cite both the original DiffMark paper and this work:

```bibtex
@article{sun2025diffmark,
  title={DiffMark: Diffusion-based Robust Watermark Against Deepfakes},
  author={Sun, Chen and Sun, Haiyang and Guo, Zhiqing and others},
  journal={Information Fusion},
  year={2025}
}

@software{diffmark_extensions,
  title={DiffMark Extensions: 10 Research Directions},
  year={2025}
}
```

---

## License

This code is provided for research purposes. Please see the original DiffMark license for more details.
