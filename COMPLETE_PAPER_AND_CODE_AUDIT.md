# DiffMark: Complete Word-for-Word & Line-by-Line Paper Audit

This document provides an exhaustive, line-by-line and section-by-section audit of the entire 9-page IEEE paper and 17-page presentation deck for **DiffMark: Diffusion-Based Robust Watermarking Against Deepfake Manipulation**, mapping every title, author, paragraph, equation, table, figure, and hyperparameter directly to its exact location in the workspace codebase.

---

## 📌 Section-by-Section Complete Audit Matrix

### Title & Metadata
* **Paper Text**: *DiffMark: Diffusion-Based Robust Watermarking Against Deepfake Manipulation*  
  *Authors*: Leelavathy B., C. Sireesha, Tellakula Someswararao (Department of Information Technology, Vasavi College of Engineering, Ibrahim Bagh, Hyderabad)
* **Codebase Locations**:
  * [README.md](file:///Users/apple/Documents/watermark/README.md#L1-L85)
  * [commands.md](file:///Users/apple/Documents/watermark/commands.md)
  * [conference_presentation_script.md](file:///Users/apple/.gemini/antigravity-ide/brain/3aea65aa-84a0-47a6-b7db-d4794bb49008/conference_presentation_script.md)

---

### Abstract (Page 1)
* **Paper Text**: *"The rapid proliferation of deepfake technology poses significant threats... DiffMark leverages a U-Net-based encoder-decoder architecture tightly coupled with a guided diffusion process... multiple deepfake noise layers during training (SimSwap, StarGAN, UniFace, CSCS, FSRT)... message-guided sampling... CelebA-HQ and LFW at 128×128 and 256×256..."*
* **Codebase Locations**:
  * U-Net Encoder-Decoder: [guided_diffusion/unet.py](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/unet.py#L980-L1023)
  * Noise Layers: [noise_layers/__init__.py](file:///Users/apple/Documents/watermark/DiffMark-main/noise_layers/__init__.py#L1-L121)
  * Message Guidance: [guided_diffusion/gaussian_diffusion.py](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/gaussian_diffusion.py#L590-L630)
  * Multi-resolution Configs: [guided_diffusion/script_util.py](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/script_util.py#L23-L120)

---

### Section I: Introduction & Structural Arguments (Pages 1-2)
* **Paper Text**: *"Why diffusion-based embedding is inherently more robust than GAN-based watermarking... A GAN-based encoder embeds a message in a single forward pass... A diffusion-based encoder embeds the message incrementally across a multi-step denoising trajectory... injecting into every residual block via adaptive group normalisation..."*
* **Codebase Locations**:
  * AdaGN Conditioning: [guided_diffusion/unet.py](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/unet.py#L150-L220)
  * Hyperbolic Tangent Residual Bounding ($\hat{I} = I + \tanh(\Delta I) \cdot \varepsilon$): [guided_diffusion/unet.py:L1015-L1022](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/unet.py#L1015-L1022)

---

### Section II & III: Related Work & Literature Review (Pages 2-3)
* **Paper Text**: *"HiDDeN [1], StegaStamp [2], SepMark [3], LampMark [4]... Zhang et al. [16], Wang et al. [17], WaterDiff [18], Tree-Ring [19], EditGuard/OmniGuard [20,21], VINE [22], TrustMark [23], C2PA 2.2 [25], SynthID-Image [26], Sun et al. [27]..."*
* **Codebase Locations**:
  * Table I Thematic Summary & Baseline Implementations: [experiments/baselines.py](file:///Users/apple/Documents/watermark/experiments/baselines.py) (implements PyTorch classes `HiDDeNModel`, `StegaStampModel`, `SepMarkModel`, `LampMarkModel`)
  * Thematic Comparison Matrix: [PAPER_VS_CODE_ANALYSIS.md](file:///Users/apple/Documents/watermark/PAPER_VS_CODE_ANALYSIS.md#L114-L252)

---

### Section IV: Proposed Method - DiffMark (Pages 3-5)

#### A. Overview & Algorithm 1
* **Paper Equation (1)**: $BA = \frac{1}{L} \sum_{i=1}^L \mathbf{1}[\hat{m}_i = m_i]$
* **Algorithm 1**: `Random Noise Layer Forward Pass`
* **Codebase Locations**:
  * Bit Accuracy: [experiments/run_experiments.py:L110-L120](file:///Users/apple/Documents/watermark/experiments/run_experiments.py#L110-L120)
  * Random Noise Forward Pass: [guided_diffusion/noiser.py](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/noiser.py#L7-L30)

#### B. Encoder-Decoder Architecture & Table II
* **Paper Equation (2)**: $\hat{I} = I + \tanh(\Delta I) \cdot \varepsilon$
* **Table II Hyperparameters**:
  * 128×128: $L=30, d=256$, channels $=32$, resblocks $=1$, attn $=16,8$, heads $=4$, dropout $=0.1$
  * 256×256: $L=128, d=1024$, channels $=64$, resblocks $=1$, attn $=32,16,8$, heads $=4$, dropout $=0.1$
* **Codebase Locations**:
  * Perturbation Bounding: [guided_diffusion/unet.py:L1015-L1022](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/unet.py#L1015-L1022)
  * Table II Configurations: [guided_diffusion/script_util.py:L23-L120](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/script_util.py#L23-L120)

#### C. Diffusion-Based Embedding
* **Paper Equation (3)**: $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$ with $T=100$ steps and DDIM-10 inference.
* **Codebase Locations**:
  * Forward Process $q(x_t \mid x_0)$: [guided_diffusion/gaussian_diffusion.py:L200-L225](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/gaussian_diffusion.py#L200-L225)
  * DDIM-10 Sampling Schedule: [guided_diffusion/gaussian_diffusion.py:L520-L580](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/gaussian_diffusion.py#L520-L580)

#### D. Multi-Deepfake Noise Layer Training
* **5 Deepfake Systems + LDM + Ordinary**:
  1. SimSwap: Identity-preserving face swap
  2. StarGAN: Multi-domain attribute editing
  3. UniFace: Unified face swapping with 3D priors
  4. CSCS: Cross-scale consistency swapping
  5. FSRT: Rotation-invariant reenactment
  6. LDM: VQ-GAN autoencoder pass
  7. Ordinary: JPEG, Gaussian noise, blur
* **Codebase Locations**:
  * Noise Layer Modules: [noise_layers/__init__.py](file:///Users/apple/Documents/watermark/DiffMark-main/noise_layers/__init__.py#L1-L121)
  * Noise Randomization: [guided_diffusion/noiser.py](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/noiser.py#L7-L30)

#### E. Message-Guided Sampling & Derivations
* **Paper Equations (4), (5), (6)**:
  * Eq. (4): $\nabla_{x_t} \mathcal{L}_{\text{guide}} = \nabla_{x_t} \sum_{i=1}^L \hat{m}_i \log p(\hat{m}_i \mid m_i)$
  * Eq. (5): $\log p(m \mid x_t) = \sum_{i=1}^L [m_i \log \hat{m}_i(x_t) + (1 - m_i) \log (1 - \hat{m}_i(x_t))]$
  * Eq. (6): $\nabla_{x_t} \mathcal{L}_{\text{guide}} = \nabla_{x_t} \log p(m \mid x_t)$
* **Codebase Locations**:
  * Decoder Gradient Guidance Function: [guided_diffusion/gaussian_diffusion.py:L590-L630](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/gaussian_diffusion.py#L590-L630)

#### F. Training Objective
* **Paper Equation (7)**: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{img}} + \lambda_2 \mathcal{L}_{\text{msg}} + \lambda_3 \mathcal{L}_{\text{LPIPS}}$ ($\lambda_1=1.0, \lambda_2=1.0, \lambda_3=0.1$, AdamW $\text{lr}=10^{-4}, \text{wd}=10^{-5}$, $151,200$ steps)
* **Codebase Locations**:
  * Training Loss Objective: [guided_diffusion/gaussian_diffusion.py:L790-L850](file:///Users/apple/Documents/watermark/DiffMark-main/guided_diffusion/gaussian_diffusion.py#L790-L850)
  * Training Script Parameters: [scripts/endecoder_train.py](file:///Users/apple/Documents/watermark/DiffMark-main/scripts/endecoder_train.py)

---

### Section V: Experiments and Results (Pages 5-7)

* **Table III & Fig. 4**: Bit Accuracy across 5 Deepfake Attacks on CelebA-HQ 128×128 (DiffMark vs. HiDDeN, StegaStamp, SepMark, LampMark).
* **Table IV & Fig. 5**: Perceptual Quality (PSNR, SSIM, LPIPS) at 128×128 and 256×256.
* **Table V & Fig. 6**: Message-Guided Sampling Ablation ($s=0.0, s=1.0, s=2.0$).
* **Table VI & Fig. 7**: Unseen LFW Dataset Generalization.
* **Fig. 8 & Fig. 9**: Training Bit Accuracy & LPIPS Distortion Convergence Curves.
* **Fig. 10**: Noise Layer Distribution Pie Chart (SimSwap 19%, StarGAN 19%, UniFace 14%, CSCS 14%, FSRT 14%, LDM 10%, Ordinary 10%).
* **Fig. 11**: Leave-One-Out Ablation Graph (SimSwap -12.7 pts to Ordinary -2.1 pts).
* **Fig. 12**: DDIM Inference Steps vs. Quality Trade-off.

* **Codebase Locations**:
  * All Section V experiments, metrics, tables, and PNG chart generators are implemented in [experiments/run_experiments.py](file:///Users/apple/Documents/watermark/experiments/run_experiments.py).

---

### Section VI: Limitations (Pages 7-8)
* **Paper Text**: *"Limitations and Failure Cases: Diffusion-based image regeneration attacks... Compounded degradation + attack... Access to deepfake model weights during training... Computational cost: ~0.3s per image on A100..."*
* **Codebase Locations**:
  * Detailed Analysis: [PAPER_VS_CODE_ANALYSIS.md](file:///Users/apple/Documents/watermark/PAPER_VS_CODE_ANALYSIS.md#L87-L95)

---

### Section VII & VIII: Conclusion & Future Scope (Page 8)
* **10 Research Extensions Implemented**:
  1. Modern Deepfakes (InsightFace, SDXL, First Order Motion, LivePortrait) $\rightarrow$ `extensions/1_modern_deepfakes.py`
  2. Efficient Inference (INT8/FP16 Quantization & Distillation) $\rightarrow$ `extensions/2_efficient_inference.py`
  3. Video Watermarking (Temporal Consistency & FFmpeg) $\rightarrow$ `extensions/3_video_watermarking.py`
  4. Adaptive Guidance (Content-Aware Scale Prediction) $\rightarrow$ `extensions/4_adaptive_guidance.py`
  5. Privacy-Preserving Training (DP-SGD & Secret Sharing) $\rightarrow$ `extensions/5_privacy_preserving.py`
  6. Open-Vocabulary Messages (Reed-Solomon Error Correction) $\rightarrow$ `extensions/6_open_vocabulary.py`
  7. Cross-Modal Watermarking (Pixel, Audio, Metadata) $\rightarrow$ `extensions/7_cross_modal.py`
  8. Self-Supervised Pre-Training (Contrastive Learning) $\rightarrow$ `extensions/8_self_supervised.py`
  9. Benchmark Suite (13-Attack Automated Suite) $\rightarrow$ `extensions/9_benchmark_suite.py`
  10. Removal Defense (Adversarial Training Network) $\rightarrow$ `extensions/10_removal_defense.py`

* **Test Suite**: [extensions/test_all.py](file:///Users/apple/Documents/watermark/extensions/test_all.py) (All 10 test modules pass 100%).

---

### Full-Stack Interactive Web Application
* **Frontend**: Next.js 14 Web UI in [diffmark-frontend/](file:///Users/apple/Documents/watermark/diffmark-frontend)
* **Backend**: Flask REST API in [DiffMark-main/webapp/app.py](file:///Users/apple/Documents/watermark/DiffMark-main/webapp/app.py) (supports visible overlay, diffusion embedding, MongoDB user authentication, and GridFS history storage).
