# DiffMark: Complete Command Reference

This file contains all commands to setup, run, test, and deploy the DiffMark framework, experiment suite, research extensions, and full-stack web application.

---

## 🚀 1. Environment Setup

### Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Install Dependencies (if setting up fresh environment)
```bash
pip install -r DiffMark-main/environment.txt
pip install lpips torchmetrics torchvision matplotlib tqdm pyyaml
```

---

## 📊 2. Run Paper Experiments (Section V & Presentation Slides)

Runs all quantitative evaluations and generates paper figures (`fig4_bit_accuracy_attacks.png`, `fig6_guidance_scale.png`, `fig10_noise_layer_distribution.png`, `fig11_leave_one_out.png`) and structured data (`results.yaml`, `results.json`).

### Standard Execution (Untrained / Initialized Architecture)
```bash
python3 experiments/run_experiments.py
```

### Execution with Trained Checkpoint
```bash
python3 experiments/run_experiments.py --checkpoint /path/to/your/checkpoint.pt --image_size 128 --batch_size 4
```

---

## 🧪 3. Research Extensions Suite (10 Extensions)

Runs automated testing across all 10 research extensions (Modern Deepfakes, Quantization/Distillation, Video Watermarking, Adaptive Guidance, Privacy Preserving, Open Vocabulary, Cross-Modal, Self-Supervised Pre-Training, Benchmark Suite, Removal Defense).

```bash
cd extensions && python3 test_all.py
```

---

## 🌐 4. Full-Stack Web Application

### Start Backend API (Flask Server on Port 5055)
```bash
cd DiffMark-main/webapp
python3 app.py
```

### Start Frontend Application (Next.js App on Port 3000)
```bash
cd diffmark-frontend
npm install
npm run dev
```

---

## 🏋️ 5. Model Training Commands

### Train DiffMark on 128×128 Resolution
```bash
cd DiffMark-main
python3 scripts/endecoder_train.py --data_dir data/celeba --image_size 128 --message_length 30 --lr 1e-4 --batch_size 4
```

### Train DiffMark on 256×256 Resolution
```bash
cd DiffMark-main
python3 scripts/endecoder_train.py --data_dir data/celeba --image_size 256 --message_length 128 --lr 1e-4 --batch_size 2
```

---

## 🔍 6. Verification & Sampling Tests

### Single Image Sampling / Embedding
```bash
cd DiffMark-main
python3 scripts/endecoder_sample.py --model_path /path/to/checkpoint.pt --image_path sample.png --output_dir results/
```
