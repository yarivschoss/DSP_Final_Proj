# 🎧 Digital Signal Processing – Final Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![DSP](https://img.shields.io/badge/Domain-DSP-ff69b4)
![Repo size](https://img.shields.io/github/repo-size/yarivschoss/DSP_Final_Proj)
![Last commit](https://img.shields.io/github/last-commit/yarivschoss/DSP_Final_Proj)

> **Author:** [Yariv Shossberger](mailto:yarivshossberger@gmail.com)  
> **Course:** Digital Signal Processing (Afeka College, 2025)

---

## 📑 Table of Contents
1. [Overview](#-overview)  
2. [Project Goals](#-project-goals)  
3. [Directory Structure](#-directory-structure)  
4. [Quick Start](#-quick-start)  
5. [Examples & Results](#-examples--results)  
6. [Technologies](#technologies)  
7. [Author](#author)   

---

## 🎯 Overview
This project demonstrates practical DSP techniques—​convolution, Butterworth filter design, and frequency-domain analysis—implemented in **Python**. The repo includes clean, modular code plus visualisations that make core DSP concepts easy to understand.

---

## 🚀 Project Goals
- **Implement** discrete-time convolution from scratch.  
- **Design & analyse** Butterworth low-pass and band-pass filters.  
- **Visualise** impulse, step, and frequency responses.  
- **Explore** group delay and linear vs. non-linear phase behaviours.  

---

## 📂 Directory Structure
```text
FinalProj/
├── src/           # Python source code
│   ├── main.py
│   ├── filters.py
│   └── utils.py
├── data/          # Input / output signal samples
├── plots/         # Auto-generated figures
└── README.md      # You are here 🙂
```

---

## 🔧 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yarivschoss/DSP_Final_Proj.git
cd DSP_Final_Proj/FinalProj
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt          # if you have one
# or minimal:
pip install numpy scipy matplotlib
```

### 4. Run the project
```bash
python src/main.py
```

Outputs (plots + CSV files) will appear in `plots/` and `data/`.

---

## 📊 Examples & Results
| Figure | Description |
| ------ | ----------- |
| ![freq resp](plots/freq_response.png) | **Frequency response** of designed Butterworth filter |
| ![imp resp](plots/imp_response.png)  | **Impulse response** vs. ideal response |
| ![conv](plots/convolution_demo.png)  | **Convolution demo** of input signal with filter kernel |

---

## 🛠️ Technologies
| Purpose                | Tool / Library |
|------------------------|----------------|
| Core language          | **Python 3.10+** |
| Numerical computing    | **NumPy**, **SciPy**  |
| Plotting               | **Matplotlib** |

---

## ✏️ Author
**Yariv Shossberger** – DSP enthusiast, hardware engineer, and EE undergrad (Afeka College).  
- ✉️ [yarivshossberger@gmail.com](mailto:yarivshossberger@gmail.com)  
- 💼 [LinkedIn](https://www.linkedin.com/in/yariv-shossberger-2334911b0)  

---

> © 2025 Yariv Shossberger — Feel free to fork, star, and improve. Contributions & feedback welcome!
