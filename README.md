# 🚘 KDP-AD  

<div align="center">

### A Knowledge-Driven Diffusion Policy for End-to-End Autonomous Driving Based on Expert Routing

[![arXiv](https://img.shields.io/badge/arXiv-todo.todo-b31b1b.svg)](https://perfectxu88.github.io/KDP-AD/) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://perfectxu88.github.io/KDP-AD/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

[**📄 Paper**](https://arxiv.org/abs/todo.todo) | [**🌐 Project Page**](https://perfectxu88.github.io/KDP-AD/)

</div>


---

## 📌 Framework

<div align="center">
<img src="./docs/assets/images/Fig_framework_white.png" width="85%" alt="KDP-AD Framework Overview">
</div>

---

## ✨ Highlights

- **Generative Policy Learning**  
  Driving modeled as conditional denoising of trajectories → captures **multi-modal behaviors** & **long-horizon consistency**.
- **Knowledge-Driven Expert Routing**  
  Sparse MoE experts encode modular driving knowledge → dynamically compose experts per scenario for **adaptive policy execution**.
- **Scalable & Interpretable**  
  Experts exhibit **structured specialization** and **cross-scenario reuse**.

---

## 🎥 Demo Video  

https://github.com/user-attachments/assets/d71d7a71-549f-40a3-b2ad-084ebd82e6e3

---

## ✅ Roadmap

- [ ] 📝 **Release Paper**  
- [ ] 💻 **Release Code**  
- [ ] 🔧 **Release Model Checkpoints** 

---

## ⚡ Getting Started

### 🔹 Installation

```bash
# Clone the repository
git clone https://github.com/PerfectXu88/KDP-AD.git
cd KDP-AD
conda create -n kdp python=3.10
conda activate kdp
# Install dependencies
pip install -r requirements.txt
```

### 🔹 Data Collection

```bash
python data_collect.py
```

### 🔹 Training

```bash
python train.py
```

### 🔹 Inference

```bash
python eval.py
```

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE). 