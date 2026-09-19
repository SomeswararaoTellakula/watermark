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

## Render Deployment

Render deploys the backend and frontend as separate web services. The included
`render.yaml` is a Blueprint configuration for both services. It pins the
backend to Python 3.11.9 because the default Render Python runtime can move
ahead of the versions supported by the backend dependencies.

For a manually created backend service, use:

- Root directory: `DiffMark-main/webapp`
- Build command: `python -m pip install --upgrade pip && python -m pip install -r requirements.txt`
- Start command: `PORT=$PORT python app.py`
- Environment variable: `PYTHON_VERSION=3.11.9`

For the frontend service, use `diffmark-frontend` as the root directory and
`npm install && npm run build` as the build command. Set
`NEXT_PUBLIC_API_URL` to the deployed backend URL followed by `/api`.

After deploying, verify the backend at `/api/health` before testing login or
watermark operations. If a build selects Python 3.14, set `PYTHON_VERSION` to
`3.11.9` and redeploy.

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

