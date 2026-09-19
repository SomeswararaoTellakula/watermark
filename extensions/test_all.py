#!/usr/bin/env python3
"""
Test script to verify all DiffMark research extensions
This script tests all 10 research extensions implemented in the extensions/ directory
"""

import torch
import numpy as np
from PIL import Image
import io

print("=" * 80)
print("Testing DiffMark Extensions")
print("=" * 80)
print()

# Test 1: Modern Deepfake Noise Layers
print("1. Testing Modern Deepfake Noise Layers...")
try:
    from extension1_modern_deepfakes import ModernDeepfakeNoiseLayer
    noise_layer = ModernDeepfakeNoiseLayer()
    dummy_input = torch.randn(1, 3, 128, 128)
    dummy_cover = dummy_input.clone()
    output = noise_layer([dummy_input, dummy_cover])
    print("   ✅ Success!")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Efficient Real-Time Inference
print("2. Testing Efficient Real-Time Inference...")
try:
    from extension2_efficient_inference import EfficientInference, DistilledDiffMark
    # Create minimal dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x, msg):
            return x, torch.randn_like(msg)
    dummy_model = DummyModel()
    pipeline = EfficientInference(dummy_model)
    print("   ✅ Success!")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Video Watermarking Extension
print("3. Testing Video Watermarking Extension...")
try:
    from extension3_video_watermarking import VideoWatermarker
    watermarker = VideoWatermarker()
    print("   ✅ Import success")
    
    # Test with dummy video
    dummy_frames = [Image.new('RGB', (128, 128), color='red') for _ in range(10)]
    watermarked_frames = watermarker.watermark_frames(dummy_frames, "test_watermark")
    print("   ✅ Initialization success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Adaptive Message-Guided Sampling
print("4. Testing Adaptive Message-Guided Sampling...")
try:
    from extension4_adaptive_guidance import AdaptiveGuide
    guide = AdaptiveGuide()
    print("   ✅ Import success")
    # Use PIL image for predict_guidance_scale
    dummy_img = Image.new('RGB', (128, 128), color='blue')
    guide_scale = guide.predict_guidance_scale(dummy_img)
    print(f"   ✅ Initialization success (scale: {guide_scale:.4f})")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 5: Privacy-Preserving Training
print("5. Testing Privacy-Preserving Training...")
try:
    from extension5_privacy_preserving import PrivacyTrainer
    trainer = PrivacyTrainer()
    print("   ✅ Import success")
    print("   ✅ Initialization success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 6: Open-Vocabulary Messages
print("6. Testing Open-Vocabulary Messages...")
try:
    from extension6_open_vocabulary import OpenVocabularyWatermark
    watermark = OpenVocabularyWatermark()
    print("   ✅ Import success")
    # Use message_codec's encode_text
    msg = watermark.message_codec.encode_text("Secret message")
    decoded = watermark.message_codec.decode_text(msg)
    print("   ✅ Initialization success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 7: Cross-Modal Watermarking
print("7. Testing Cross-Modal Watermarking...")
try:
    from extension7_cross_modal import CrossModalWatermarker
    watermarker = CrossModalWatermarker()
    print("   ✅ Import success")
    print("   ✅ Initialization success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 8: Self-Supervised Pre-Training
print("8. Testing Self-Supervised Pre-Training...")
try:
    from extension8_self_supervised import SelfSupervisedPretrainer
    pretrainer = SelfSupervisedPretrainer()
    print("   ✅ Import success")
    print("   ✅ Initialization success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 9: Benchmark Suite
print("9. Testing Benchmark Suite...")
try:
    from extension9_benchmark import DiffMarkBenchmark
    benchmark = DiffMarkBenchmark()
    print("   ✅ Import success")
    print(f"   ✅ Number of attacks: {len(benchmark.attacks)}")
    print("   ✅ Initialization success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 10: Watermark Removal Defense
print("10. Testing Watermark Removal Defense...")
try:
    from extension10_removal_defense import RemovalDefense
    defense = RemovalDefense()
    print("   ✅ Import success")
    print("   ✅ Initialization success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("All extension tests completed!")
print("=" * 80)
