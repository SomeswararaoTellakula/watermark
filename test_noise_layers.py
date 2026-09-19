#!/usr/bin/env python3
"""
Comprehensive Experimental Test Suite for DiffMark Noise Layers
Tests: SimSwap, FSRT, CSCS, StarGAN, UniFace + Ordinary Distortions (GaussianNoise, Blur, JPEG, etc.)
"""
import os
import sys
import time
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DiffMark-main'))

from guided_diffusion import dist_util, logger

print("=" * 80)
print("DIFFMARK NOISE LAYER EXPERIMENTAL TEST SUITE")
print("=" * 80)
print()

device = dist_util.dev()
print(f"Using device: {device}")
print()


def make_dummy_batch(batch_size=2, image_size=128):
    x = torch.randn(batch_size, 3, image_size, image_size, device=device).clamp(-1, 1)
    cover = x.clone().roll(1, dims=0)
    return x, cover


def test_layer(name, layer_class, x, cover, verbose=True):
    t0 = time.time()
    try:
        layer = layer_class().to(device).eval()
        output = layer([x, cover])
        elapsed = time.time() - t0
        shape_ok = tuple(output.shape) == tuple(x.shape)
        range_ok = output.min().item() >= -2.0 and output.max().item() <= 2.0
        has_nan = torch.isnan(output).any().item()
        status = "PASS" if (shape_ok and range_ok and not has_nan) else "FAIL"
        detail = f"shape={tuple(output.shape)} range=[{output.min().item():.2f},{output.max().item():.2f}] nan={has_nan}"
        if verbose:
            icon = "✅" if status == "PASS" else "❌"
            print(f"   {icon} {name:<20} | {status:<4} | {elapsed*1000:>7.1f} ms | {detail}")
        return status == "PASS", elapsed, detail
    except Exception as e:
        elapsed = time.time() - t0
        if verbose:
            print(f"   ❌ {name:<20} | ERR  | {elapsed*1000:>7.1f} ms | Exception: {type(e).__name__}: {str(e)[:100]}")
        return False, elapsed, f"Exception: {e}"


print("=" * 80)
print("TEST 1: DEEPFAKE SWAP / REENACTMENT LAYERS")
print("=" * 80)
from noise_layers import SimSwap, StarGAN, UniFace, CSCS, FSRT

x_128, cover_128 = make_dummy_batch(2, 128)
results = {}
for name, cls in [("SimSwap (face swap)", SimSwap),
                  ("StarGAN (attr edit)", StarGAN),
                  ("UniFace (3D swap)", UniFace),
                  ("CSCS (cross-scale)", CSCS),
                  ("FSRT (reenactment)", FSRT)]:
    ok, t, d = test_layer(name, cls, x_128, cover_128)
    results[name] = ok

print()
print("=" * 80)
print("TEST 2: ORDINARY DISTORTION LAYERS (128x128)")
print("=" * 80)
from noise_layers.ordinary.test_noise import (
    Identity, Resize, Dropout, GaussianNoise, SaltPepper,
    GaussianBlur, MedianBlur, Brightness, Contrast, Saturation, Hue, JpegTest
)

ordinary_layers = [
    ("Identity (passthru)", Identity, {}),
    ("Resize (0.8x)", Resize, {}),
    ("Dropout p=0.6", Dropout, {}),
    ("GaussianNoise", GaussianNoise, {}),
    ("SaltPepper p=0.1", SaltPepper, {}),
    ("GaussianBlur", GaussianBlur, {}),
    ("MedianBlur", MedianBlur, {}),
    ("Brightness", Brightness, {}),
    ("Contrast", Contrast, {}),
    ("Saturation", Saturation, {}),
    ("Hue", Hue, {}),
    ("JpegTest Q=50", JpegTest, {}),
]
for name, cls, kwargs in ordinary_layers:
    ok, t, d = test_layer(name, cls, x_128, cover_128)
    results[name] = ok

print()
print("=" * 80)
print("TEST 3: MULTI-RESOLUTION (256x256) COMPATIBILITY")
print("=" * 80)
x_256, cover_256 = make_dummy_batch(2, 256)
multi_res = [
    ("SimSwap @256", SimSwap),
    ("StarGAN @256", StarGAN),
    ("CSCS @256", CSCS),
    ("GaussianNoise @256", GaussianNoise),
    ("JpegTest @256", JpegTest),
    ("Resize(0.8x) @256", Resize),
]
for name, cls in multi_res:
    ok, t, d = test_layer(name, cls, x_256, cover_256)
    results[name] = ok

print()
print("=" * 80)
print("TEST 4: RANDOM NOISE SAMPLER (Paper Training Configuration)")
print("=" * 80)
try:
    from guided_diffusion.noiser import Random_Noise
    sampler = Random_Noise(['SimSwap', 'StarGAN', 'UniFace', 'CSCS', 'FSRT', 'LDM', 'Ordinary'])
    for res_name, xb, cb in [("128", x_128, cover_128), ("256", x_256, cover_256)]:
        t0 = time.time()
        out = sampler([xb, cb])
        elapsed = (time.time() - t0) * 1000
        shape_ok = tuple(out.shape) == tuple(xb.shape)
        ok = shape_ok and not torch.isnan(out).any().item()
        icon = "✅" if ok else "❌"
        print(f"   {icon} Random_Noise [{res_name:>3}p] | {elapsed:>7.1f} ms | shape={tuple(out.shape)} range=[{out.min().item():.2f},{out.max().item():.2f}]")
        results[f"Random_Noise@{res_name}"] = ok
except Exception as e:
    print(f"   ❌ Random_Noise sampler | Exception: {type(e).__name__}: {e}")
    results["Random_Noise"] = False

print()
print("=" * 80)
print("TEST 5: SHAPE & VALUE SANITY ACROSS BATCH SIZES (1, 4, 8)")
print("=" * 80)
for bs in [1, 4, 8]:
    xb, cb = make_dummy_batch(bs, 128)
    for name, cls in [("GaussianNoise", GaussianNoise), ("Dropout", Dropout), ("JpegTest", JpegTest)]:
        ok, t, d = test_layer(f"{name} bs={bs}", cls, xb, cb, verbose=True)
        results[f"{name}_bs{bs}"] = ok

print()
print("=" * 80)
passed = sum(1 for v in results.values() if v)
total = len(results)
print(f"SUMMARY: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
if passed == total:
    print("✅ ALL NOISE LAYER TESTS PASSED")
else:
    failed = [k for k, v in results.items() if not v]
    print(f"❌ FAILED: {failed}")
print("=" * 80)
