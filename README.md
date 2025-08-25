# ❤️ Heart Disease Prediction Pipeline

> 🫀 **Predict heart disease risk using machine learning **  
> Built as the **final project for Microsoft Sprints X** 🚀
## 🐳 Microsoft Sprints X

This project was developed as part of **Microsoft Sprints X**, a hands-on AI/ML training program designed to empower developers with real-world machine learning experience.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
## 📌 Important Note: For Learning Only

> ⚠️ **This pipeline is a learning project — NOT a medical diagnostic tool.**

- ❌ **Do not use this app to diagnose or treat heart disease**
- 🧪 It was trained on a small dataset (303 patients) for educational purposes
- 📚 Designed to teach ML concepts: preprocessing, modeling, deployment
- 🛑 Not validated for clinical use
- 📉 Performance may vary in real-world settings

> This project is part of **Microsoft Sprints X** to build foundational AI/ML skills — not to replace professional medical advice.

---
## 🏆 Final Model: Logistic Regression

✅ **Model Used**: `Logistic Regression`  
📊 **F1-Score**: `0.877`  
🎯 **AUC**: `0.951`  
🔍 **Why?** Logistic Regression provides stable, interpretable results with excellent balance of precision and recall.

> This model was selected as the final pipeline due to its high F1-score and reliability on imbalanced medical data.

---
## 🎓 Skills demonstrated:
- End-to-end ML pipeline
- Model interpretability
- Production-ready deployment
- UI/UX for healthcare applications

---

## 🎯 Project Overview

This **Heart Disease Prediction Pipeline** is a full-stack machine learning system that uses clinical data to predict the presence of heart disease with high accuracy.

Built during **Microsoft Sprints X**, this project demonstrates end-to-end ML development:
- ✅ Data preprocessing & EDA
- ✅ Feature selection & dimensionality reduction
- ✅ Supervised & unsupervised learning
- ✅ Model optimization & deployment
- ✅ Interactive UI with real-time predictions


## 🧰 Tech Stack

| Tool | Purpose |
|------|--------|
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="20"> **Python** | Core language |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikit-learn-original.svg" width="20"> **Scikit-learn** | ML models & pipelines |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" width="20"> **Pandas / NumPy** | Data manipulation |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" width="20"> **Matplotlib / Seaborn** | Visualization |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/streamlit/streamlit-original.svg" width="20"> **Streamlit** | Interactive UI |
| 🐍 **Random Forest, SVM, RFE, PCA** | Advanced ML techniques |
| ☁️ **Streamlit Community Cloud** | Deployment |

---

## 📦 Project Structure

```
heart-disease-predictor/
├── ui/
│   └── app.py               # 🖥️ Streamlit interface
├── models/
│   └── final_model.pkl      # 🧠 Trained ML pipeline
├── custom_transformers.py   # 🔧 Custom feature selector
├── requirements.txt         # 📦 Dependencies
├── README.md                # 📄 This file
└── .gitignore
```

---



🎯 Key features:
- **oldpeak**, **thalach**, **cp_3**, and **ca_2.0** are key predictors
- Real-time risk scoring with progress bar
- Clinically validated input ranges

---

## 🚀 How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\Activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run ui/app.py
```

---





## 🙌 Acknowledgments

- Dataset: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Built with ❤️ using [Streamlit](https://streamlit.io)
- Special thanks to Microsoft Sprints X mentors

---


