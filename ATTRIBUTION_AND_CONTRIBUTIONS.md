
# Explicit Attribution and Contribution Breakdown

## 1. Original Work (Sun et al., 2025)
**Paper**: "DiffMark: Diffusion-Based Robust Watermark Against Deepfakes"  
**Authors**: Chen Sun, Haiyang Sun, Zhiqing Guo, Yunfeng Diao, Liejun Wang, Dan Ma, Gaobo Yang, Keqin Li  
**Venue**: Information Fusion, 2025  
**Code Repository**: [https://github.com/vpsg-research/DiffMark](https://github.com/vpsg-research/DiffMark)

### What is included from the original work in this project:
- All files in `/DiffMark-main/`:
  - Guided diffusion model code (unet.py, gaussian_diffusion.py, etc.)
  - Noise layer implementations (SimSwap, StarGAN, UniFace, CSCS, FSRT, LDM, Ordinary)
  - Training and evaluation scripts (scripts/image_train.py, scripts/image_test.py)
  - The original PDF: `DiffMark-main/diffmark_paper_v2.pdf`
  - Configuration files, README, environment.txt
  - Results logs and sample data in `/DiffMark-main/results/` and `/DiffMark-main/data/`

---

## 2. Our New, Original Contributions
Everything **outside** of `/DiffMark-main/` is our new, original work:

### A. Full-Stack Web Application
- `/DiffMark-main/webapp/`: Flask backend API
  - User authentication (signup/login)
  - Watermark embedding/verification endpoints
  - Image processing endpoints (EDA, clean, normalize, augment)
  - MongoDB + GridFS integration
- `/diffmark-frontend/`: Next.js + React + Tailwind CSS frontend
  - Modern UI with motion animations
  - User dashboard and history
  - Image/video watermarking interface

### B. 10 Research Extensions (in `/extensions/`)
1. Modern Deepfake Noise Layers (extension1)
2. Efficient Real-Time Inference (extension2)
3. Video Watermarking Extension (extension3)
4. Adaptive Message-Guided Sampling (extension4)
5. Privacy-Preserving Training (extension5)
6. Open-Vocabulary Messages (extension6)
7. Cross-Modal Watermarking (extension7)
8. Self-Supervised Pre-Training (extension8)
9. Benchmark Suite (extension9)
10. Defense Against Watermark Removal Attacks (extension10)

### C. Paper Revision and Reviewer Response
- `diffmark_paper_revised.tex`: Revised version with new experiments and content
- `RESPONSE_TO_REVIEWERS.md`: Point-by-point response to reviewers
- `DIFFMARK_IMPLEMENTATION_GUIDE.md`: Implementation guide

---

## 3. Citation Guidelines
If you use this project, **you must cite the original DiffMark paper**:

```bibtex
@article{SUN2025103801,
  title = {DiffMark: Diffusion-based Robust Watermark Against Deepfakes},
  author = {Chen Sun and Haiyang Sun and Zhiqing Guo and Yunfeng Diao and Liejun Wang and Dan Ma and Gaobo Yang and Keqin Li},
  journal = {Information Fusion},
  year = {2025},
}
```

If you use our new contributions (web app, extensions, revised paper), please consider citing our work as well (if/when published).

---

## 4. Honesty Statement
We explicitly acknowledge that:
- The core DiffMark algorithm, model architecture, and deepfake noise layers are the work of Sun et al. (2025)
- Our additions are clearly separated in the directory structure
- We have not plagiarized or misrepresented any work
- All claims of novelty in our revised paper apply only to our new contributions
