# DiffMark Full-Stack Implementation & Research Extensions

This repository contains:
- The original DiffMark implementation by Sun et al. (2025)
- 10 novel research extensions
- A full-stack web application for watermarking and verification
- Comprehensive experiment scripts

## Project Structure

```
watermark/
├── DiffMark-main/          # Original DiffMark implementation
├── diffmark-frontend/      # Next.js frontend
├── extensions/             # 10 research extensions
├── experiments/            # Experiment scripts
├── PAPER_VS_CODE_ANALYSIS.md # Detailed comparison
└── README.md               # This file
```

## Quickstart

### Web Application

#### Backend (Flask)
```bash
cd DiffMark-main/webapp
python app.py
```
The backend runs on port 5055.

#### Frontend (Next.js)
```bash
cd diffmark-frontend
npm install
npm run dev
```
The frontend runs on port 3000.

### Research Extensions

To test all 10 research extensions:
```bash
cd extensions
python3 test_all.py
```

### Experiments

To run the paper experiments:
```bash
cd experiments
python3 run_experiments.py
```

## Extensions

The 10 research extensions are:
1. Modern Deepfake Noise Layers
2. Efficient Real-Time Inference
3. Video Watermarking
4. Adaptive Message-Guided Sampling
5. Privacy-Preserving Training
6. Open-Vocabulary Messages
7. Cross-Modal Watermarking
8. Self-Supervised Pre-Training
9. Benchmark Suite
10. Removal Defense

## Paper vs Code

For a detailed comparison of the paper claims vs actual code, see `PAPER_VS_CODE_ANALYSIS.md`.

