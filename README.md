<h1 align="center">AdaptiveGuard Replication Package</h1>

<h4 align="center">This repository provides the replication package for our AdaptiveGuard experiments on continual LLM safety alignment. </h4>

<p align="center">
  <img src="./imgs/adaptiveguard_logo.png" alt="AdaptiveGuard Logo" width="150"/>
</p>

## 🛡️ Continual Guardrail — AdaptiveGuard

<p align="center">
  <img src="./imgs/adaptiveguard_overview.png" alt="AdaptiveGuard Overview" style="width:100%; max-width:1000px;"/>
</p>

---

## 📚 Table of Contents

1. [Environment Setup](#1-environment-setup)  
2. [Reproduce RQ1](#2-reproduce-rq1)  
3. [Reproduce RQ2](#3-reproduce-rq2)
4. [Reproduce RQ3](#4-reproduce-rq3)
5. [Citation](#5-citation)

---

## 1. Environment Setup

We recommend using **Python 3.12** for best compatibility and performance.

### Step 1: Install Python Requirements

To install all necessary dependencies, run:

```bash
pip install -r requirements.txt
```

### Step 2: Install PyTorch with CUDA

If you’re using an NVIDIA GPU, we highly recommend installing PyTorch with CUDA support to accelerate training and inference.
Follow the official installation guide from PyTorch:
👉 https://pytorch.org/get-started/locally

---

## 2. Reproduce RQ1
### (RQ1) How effective is our AdaptiveGuard approach in identifying unknown jailbreak prompts?
<p align="center">
  <img src="./imgs/RQ1_ood_detection_performance.png" alt="RQ1 Results" style="width:100%; max-width:1000px;"/>
</p>

TODO

---

## 3. Reproduce RQ2
### (RQ2) How effective is our AdaptiveGuard approach at defending against unknown jailbreak attacks when continuously updated through detected OOD prompts?
<p align="center">
  <img src="./imgs/RQ2_DSR_Comparison.png" alt="RQ2 Results" style="width:100%; max-width:1000px;"/>
</p>

TODO

---

## 4. Reproduce RQ3
### (RQ3) How much does our AdaptiveGuard approach forget original in-distribution prompts after continuous updates with detected OOD prompts?
<p align="center">
  <img src="./imgs/RQ3_F1_Comparison.png" alt="RQ3 Results" style="width:100%; max-width:1000px;"/>
</p>

TODO

---

## 5. Citation

```bibtex
under review
```
