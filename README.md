# 🏡 House Price Predictor – ML Web App

This Machine Learning project predicts house prices based on real-world housing data using **Linear Regression**, **Ridge**, and **Lasso** models. Built with **Python**, deployed with **Streamlit**, and based on the **Ames Housing dataset**.

---

![App Banner](assets/banner.png)

## 🔍 Overview

This project demonstrates:
- End-to-End ML workflow for regression
- Regularization techniques to avoid overfitting
- Building a UI with Streamlit for user interaction
- Saving and loading models using `pickle`

---

## 📁 Dataset

- Source: [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- Features used:
  - Lot Area
  - Year Built
  - Total Basement Area
  - Living Area (GrLivArea)
  - Garage Capacity
  - Overall Quality
  - Sale Price (Target)

---

## 🚀 Tech Stack

| Component       | Tech Used              |
|----------------|------------------------|
| Language        | Python 🐍             |
| ML Models       | Linear, Ridge, Lasso   |
| Data Wrangling  | Pandas, NumPy          |
| Visualization   | Matplotlib, Seaborn    |
| Deployment      | Streamlit Cloud         |
| Others          | Pickle, Scikit-learn   |

---

## 💡 App Preview

<img src="https://housepricepredictorbynarasimhamanam.streamlit.app/" alt="House Price Predictor App" width="800"/>

---

## 🧠 How it Works

1. 📦 Load Ames Housing dataset  
2. 🧼 Clean and select important features  
3. ⚙️ Train 3 regression models  
4. 📈 Evaluate using R² and RMSE  
5. 💾 Save best model as `house_price_model.pkl`  
6. 🌐 Deploy with Streamlit

---

## 🛠️ Running the App Locally

### ✅ Step 1: Clone the repo

```bash
git clone https://github.com/<your-username>/house-price-predictor.git
cd house-price-predictor
