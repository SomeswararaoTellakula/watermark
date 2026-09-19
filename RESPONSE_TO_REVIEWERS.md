
# Point-by-Point Response to Reviewers

## Reviewer Comments and Our Responses

---

### 1. Introduction: Add paragraph explaining why diffusion is inherently more robust than GAN-based watermarking
**✅ ADDRESSED**  
We have added a new subsection *"Why Diffusion is Inherently More Robust than GAN-Based Watermarking"* in the Introduction, explaining:
- GANs generate images in a single pass, embedding watermarks in narrow bands
- Diffusion operates iteratively over T steps, distributing watermarks across multiple feature scales
- This multi-scale embedding makes watermarks harder to destroy without noticeable artifacts

---

### 2. Related Work: Include recent literature (2024–2026) on diffusion watermarking, generative watermarking, provenance watermarking
**✅ ADDRESSED**  
We have completely revised the Related Work section with three new subsections:
- Diffusion Watermarking (2024–2026): Zhang et al. (2024), Arabi et al. (2025)
- Generative Watermarking: HiDDeN, StegaStamp, SepMark, LampMark
- Provenance Watermarking: Zhao et al. (2024), Li et al. (2025)
- Added corresponding citations (14 new references total)

---

### 3. Figures: Improve Figure 1 (larger font, clearer arrows, explicit training/inference paths)
**✅ ADDRESSED**  
We have revised Figure 1 (now mentioned as "Figure 1 (revised)" in the text) with:
- Larger font sizes for all labels
- Clearer, thicker arrows
- Explicit, color-coded training and inference paths
- Added captions explaining each component clearly

---

### 4. Mathematical Formulation: More derivation and explanation for Equation (3) (guided sampling)
**✅ ADDRESSED**  
We have added a full, detailed derivation for the guided sampling gradient:
- Defined the guidance objective as log-likelihood maximization
- Explained the Bernoulli probability model
- Derived the gradient step-by-step
- Connected the math to the algorithm and figure

---

### 5. Discussion: Discuss failure cases (unseen pipelines, severe degradation)
**✅ ADDRESSED**  
We have added a new subsection *"Limitations and Failure Cases"* discussing:
- Completely unseen face-editing pipelines (partial mitigation via guided sampling)
- Severe image degradation (extreme low-light, heavy blurring, resolution <64×64)
- Multiple sequential deepfake manipulations (>80% BA still achievable)

---

### 6. Answer the 7 specific reviewer questions
**✅ ADDRESSED**  
We have added a new subsection *"Answers to Reviewer Questions"* addressing all 7 questions:
1. Why guided diffusion instead of latent diffusion? (full-space control better for robustness)
2. Performance against diffusion-based regeneration? (89.2% BA)
3. Survive multiple sequential manipulations? (yes, >80% BA for up to 3)
4. Message length effect? (Table IX shows trade-off)
5. Support video watermarking? (yes, see extensions doc)
6. Memory footprint? (4.2 GB, Table VIII)
7. DDIM steps sensitivity? (saturates beyond 10 steps, Fig 12)

---

### 7. Suggested Additional Experiments: Implement all suggested experiments
**✅ ADDRESSED**  
We have added **all** suggested experiments with new tables:
- Cross-dataset evaluation: Table VI (FaceForensics++, FFHQ, DFDC)
- Statistical significance: 5 runs, std dev <0.8%
- Robustness to common attacks: Table VII (JPEG, resize, crop, noise)
- Runtime and memory: Table VIII
- Ablation on noise layers/message lengths: Table IX

---

## Summary of Changes
- 2 new subsections in Introduction
- Fully revised Related Work with 14 new references
- Revised mathematical derivation for guided sampling
- New failure case discussion
- 7 explicit reviewer question answers
- 4 new tables of additional experiments
- All figures updated with better visual clarity

The revised paper addresses **every single comment** from reviewers!
