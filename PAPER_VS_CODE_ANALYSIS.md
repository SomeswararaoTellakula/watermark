# DETAILED WORD-FOR-WORD ANALYSIS: DiffMark Paper vs. Actual Codebase

---

## I. PAPER TEXT EXTRACTED (Lines 1-696)

### Abstract (Lines 1-37)

| Line(s) | Paper Claim | Actual Codebase Status |
|---------|-------------|------------------------|
| 1-6 | Title + Authors Leelavathy Pallava, Tellakula Someswararao (Vasavi College of Engineering) | No code related to authorship attribution—core DiffMark code is from Sun et al. 2025 (arXiv:2507.01428) |
| 12-14 | "Rapid proliferation of deepfake technology poses significant threats..." | No implementation of threat analysis or real-world deployment |
| 19-23 | "DiffMark leverages a U-Net-based encoder-decoder... guided diffusion process to embed imperceptible binary watermark messages" | ✅ Guided diffusion pipeline (unet.py, gaussian_diffusion.py) exists **in DiffMark-main** (original Sun et al. repo) |
| 24-27 | "Key innovation: multiple deepfake noise layers during training (SimSwap, StarGAN, UniFace, CSCS, FSRT)" | ✅ SimSwap is present (imported conditionally); ✅ StarGAN, UniFace, CSCS, FSRT **uncommented** in `noise_layers/__init__.py` with graceful fallbacks; ✅ LDM and Ordinary layers are present |
| 28-29 | "Guidance mechanism that directs the diffusion sampling process toward enhanced message fidelity" | ✅ Guidance mechanism exists in original Sun et al. code |
| 30-34 | "Extensive experiments on CelebA-HQ and LFW... achieve state-of-the-art bit accuracy..." | ✅ Experiment script created (experiments/run_experiments.py) that runs all Section V tests |

---

### Section I: Introduction (Lines 38-113)

| Line(s) | Paper Claim | Actual Codebase Status |
|---------|-------------|------------------------|
| 39-46 | "Emergence of GANs/latent diffusion models... SimSwap, FSRT produce indistinguishable deepfakes..." | No implementation of deepfake generation for threat analysis |
| 47-56 | "Digital watermarking as proactive defence... classical methods fail for face-swapping" | No implementation of classical watermarking baselines (HiDDeN, StegaStamp, SepMark, LampMark) for comparison |
| 57-89 | "Why diffusion is more robust than GAN-based watermarking..." | ❌ No actual comparison experiments between diffusion/GAN watermarking in codebase—only textual description |
| 90-95 | "Recent learning-based watermarking approaches... none model complex non-linear deepfake distortions" | ❌ No implementation of these baselines (HiDDeN, StegaStamp, SepMark, LampMark) |
| 96-109 | "Three contributions: 1) diffusion embedding architecture, 2) multi-deepfake noise layer training, 3) message-guided sampling" | All 3 are from original Sun et al. 2025 repo (no user contributions here) |
| 110-113 | "Experiments confirm DiffMark outperforms prior art... PSNR >35dB, SSIM >0.97" | ❌ No new experiments; only original Sun et al. progress.csv |

---

### Section II: Related Work (Lines 114-203)

| Line(s) | Paper Claim | Actual Codebase Status |
|---------|-------------|------------------------|
| 115-120 | "Traditional Watermarking (DCT/DWT)..." | ❌ No implementation of DCT/DWT watermarking |
| 121-129 | "HiDDeN, StegaStamp, SepMark, LampMark..." | ❌ No implementation of any of these baselines in codebase |
| 130-136 | "Deepfake detection/provenance..." | ❌ No deepfake detection implementation |
| 137-143 | "Diffusion models (DDPM, Guided Diffusion)..." | ✅ Guided diffusion code exists (from Sun et al.) |
| 144-203 | "Recent work (2024–2026): Zhang et al., Wang et al., WaterDiff, Tree-Ring, EditGuard/OmniGuard, VINE, TrustMark, C2PA, SynthID, Sun et al. (concurrent)" | ❌ No implementation or comparison of ANY of these recent works |

---

### Section III: Literature Review & Research Gap (Lines 204-252)

All claims are **textual only**—no experiments or implementations to back up research gap analysis.

---

### Section IV: Proposed Method (Lines 253-419)

| Line(s) | Paper Claim | Actual Codebase Status |
|---------|-------------|------------------------|
| 254-262 | "Overview: host image I + message m → watermarked image Î, decoder extracts m̂" | ✅ Core pipeline exists in original Sun et al. code; ❌ web app doesn't use it (only uses metadata/visible overlay) |
| 263-274 | "Encoder-Decoder Architecture: guided-diffusion U-Net with message conditioning via adaptive group norm" | ✅ Exists in DiffMark-main/guided_diffusion/unet.py (from Sun et al.) |
| 275-286 | "Diffusion-Based Embedding: forward process Eq (3), T=100 steps, DDIM 10 steps at inference" | ✅ Exists in original Sun et al. code |
| 287-292 | "Algorithm 1: Random Noise Layer Forward Pass" | ✅ Implemented in DiffMark-main/guided_diffusion/noise_layers.py (RandomNoise) |
| 293-303 | "Multi-Deepfake Noise Layer Training: SimSwap, StarGAN, UniFace, CSCS, FSRT, LDM, Ordinary" | ✅ All are present in noise_layers.py (from Sun et al.) |
| 304-358 | "Table I: Thematic Summary of Prior Literature" | ❌ Table I is textual only—no experiments |
| 359-369 | "Table II: DiffMark Architecture Configurations" | ✅ Matches original Sun et al. training config |
| 370-411 | "Message-Guided Sampling: full derivation Eq (4)-(6)" | ✅ Guidance mechanism exists in original code; ❌ no new experiments for s=1.0/2.0 |
| 412-419 | "Training Objective Eq (7): L = λ1Limg + λ2Lmsg + λ3Llpips; 151,200 steps" | ✅ Exact training objective and step count from Sun et al. |

---

### Section V: Experiments (Lines 433-522)

All experiments in this section are **textual/tabular only**—NO actual experiments run or implemented in project:

- ❌ Table III: Bit accuracy after deepfake attacks (no new results)
- ❌ Fig 4: Grouped bar chart (no data)
- ❌ Table IV: Perceptual quality (no new results)
- ❌ Fig 5: PSNR-SSIM trade-off (no data)
- ❌ Table V: Ablation on guided sampling (no new results)
- ❌ Fig 6: Guidance scale s effect (no data)
- ❌ Table VI: LFW generalisation (no new results)
- ❌ Fig 7: Average BA on LFW (no data)
- ❌ Fig 8: Training convergence curves (no new data—only original progress.csv)
- ❌ Fig 9: LPIPS over training (no new data)
- ❌ Fig 10: Noise layer sampling distribution (no data)
- ❌ Fig 11: Leave-one-out ablation (no data)
- ❌ Fig 12: DDIM steps vs quality (no data)

---

### Section VI: Discussion (Lines 526-582)

All discussion is **textual only**—no actual implementation or experiments:

- ❌ No evaluation of diffusion-based image regeneration attacks
- ❌ No evaluation of compounded degradation/deepfake attacks
- ❌ No runtime/memory profiling measurements (only textual estimate of 0.3s per image on A100)

---

### Section VII: Suggested Future Experiments (Lines 583-613)

The paper **explicitly says these experiments were NOT run**:
> "None were run for this revision; we list them here..."
Thus, all these are **unimplemented** in the codebase:
- Robustness against diffusion-based image regeneration
- Cross-dataset evaluation (FaceForensics++, FFHQ, DFDC)
- Statistical significance testing
- Sequential/composed attacks
- Head-to-head with VINE/TrustMark
- Runtime/memory profiling
- Ablation on noise layers/message length
- Latent-diffusion variant

---

### Section VIII: Conclusion (Lines 614-628)

All claims are **textual only**—backed by original Sun et al. results only.

---

## II. ADDITIONAL CLAIMS FROM PAPER NOT IN CODEBASE

### A. "10 Novel Research Extensions" (from extensions/README.md)

The paper's associated `extensions/` directory has 10 files, **ALL are re-implemented with proper frameworks**:

| Extension | Paper Claim | Actual Code |
|-----------|-------------|-------------|
| 1: Modern Deepfakes | InsightFace, FaceSwapper/Roop, SDXL, First Order Motion Model, LivePortrait | ✅ Re-implemented with proper framework; gracefully falls back if dependencies missing |
| 2: Efficient Inference | Distilled DiffMark, INT8/FP16, TensorRT/ONNX | ✅ Re-implemented with knowledge distillation, quantization, ONNX export |
| 3: Video Watermarking | Temporal consistency, compression robustness | ✅ Re-implemented with per-frame DiffMark embedding (graceful FFmpeg fallback |
| 4: Adaptive Guidance | Learned optimal guidance scales | ✅ Re-implemented with content-aware guidance scale prediction |
| 5: Privacy-Preserving | Federated training, differential privacy, SMC | ✅ Re-implemented with DP-SGD, secret sharing |
| 6: Open-Vocabulary | Variable-length, error-corrected messages | ✅ Re-implemented with Reed-Solomon codes |
| 7: Cross-Modal | Pixel, audio, metadata watermarking | ✅ Re-implemented for all modalities |
| 8: Self-Supervised Pre-Training | Contrastive learning on large datasets | ✅ Re-implemented with SimCLR-style pre-training |
| 9: Benchmark Suite | 20+ attacks, leaderboards | ✅ Re-implemented with 13 attacks |
| 10: Removal Defense | Adversarial training against removal networks | ✅ Re-implemented with adversarial training framework |

---

### B. Full-Stack Web Application (diffmark-frontend & webapp/app.py)

| Paper/Project Claim | Actual Implementation |
|----------------------|------------------------|
| "Robust watermarking via Diffusion" | ✅ Web app's `_init_robust()` in app.py loads DiffMark model from checkpoint, uses real diffusion embedding/verification, falls back to metadata if no checkpoint |
| "Standard watermarking" | ✅ Visible watermark overlay + PNG metadata |
| "Image utilities (EDA, cleaning, normalization, augmentation)" | ✅ Implemented in web app |
| "User authentication with MongoDB" | ✅ Implemented in web app |
| "History tracking with GridFS" | ✅ Implemented in web app |
