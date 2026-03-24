# Classroom Equipment Failure Predictor

A machine learning-based system to predict failure risk of classroom equipment such as projectors, ACs, lighting systems, and smartboards. The system integrates model lifecycle concepts like training, evaluation, and explainability into an interactive dashboard built using Streamlit.

---

## 🚀 Objective

To proactively identify equipment failure risks using historical data and operational parameters, enabling preventive maintenance and reducing downtime in smart classrooms.

---

## 🏗️ System Overview

The system consists of:

- Machine Learning Models trained on equipment-specific datasets  
- Interactive Dashboard (Streamlit) for visualization and prediction  
- Model Registry for managing trained models  
- Explainability Module to understand feature impact  
- Hybrid Prediction System combining ML with domain logic  

---

## ⚙️ Features

### 🔹 1. Overview Dashboard
- Equipment distribution  
- Failure rate analysis  
- Usage vs failure insights  
- Maintenance impact visualization  

---

### 🔹 2. Data Explorer
- Equipment-specific dataset view  
- Feature inspection  
- Structured tabular visualization  

---

### 🔹 3. Prediction Module
- Equipment selection (Projector, AC, Lighting, Smartboard)  
- Dynamic input parameters  
- Real-time failure prediction  

#### ✅ Hybrid Prediction Logic
- ML model generates base probability  
- Rule-based adjustments applied for critical conditions:
  - Ghost touch issues  
  - Poor touch responsiveness  
  - Long maintenance gaps  
  - Corrective maintenance history  

---

### 🔹 4. Explainability
- Feature importance visualization  
- Supports:
  - Logistic Regression (coefficients)  
  - Tree-based models (feature importance)  
- Interactive Plotly charts  

---

### 🔹 5. Model Evaluation (Post-Mortem)
- Model performance summary  
- Stability insights  
- Limitations and real-world considerations  

---

## 📊 Technologies Used

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- Plotly  
- Matplotlib  

---

## 📁 Project Structure
# 🧠 Smart Classroom Equipment Failure Predictor

A machine learning-based system to predict failure risk of classroom equipment such as projectors, ACs, lighting systems, and smartboards. The system integrates model lifecycle concepts like training, evaluation, and explainability into an interactive dashboard built using Streamlit.

---

## 🚀 Objective

To proactively identify equipment failure risks using historical data and operational parameters, enabling preventive maintenance and reducing downtime in smart classrooms.

---

## 🏗️ System Overview

The system consists of:

- Machine Learning Models trained on equipment-specific datasets  
- Interactive Dashboard (Streamlit) for visualization and prediction  
- Model Registry for managing trained models  
- Explainability Module to understand feature impact  
- Hybrid Prediction System combining ML with domain logic  

---

## ⚙️ Features

### 🔹 1. Overview Dashboard
- Equipment distribution  
- Failure rate analysis  
- Usage vs failure insights  
- Maintenance impact visualization  

---

### 🔹 2. Data Explorer
- Equipment-specific dataset view  
- Feature inspection  
- Structured tabular visualization  

---

### 🔹 3. Prediction Module
- Equipment selection (Projector, AC, Lighting, Smartboard)  
- Dynamic input parameters  
- Real-time failure prediction  

#### ✅ Hybrid Prediction Logic
- ML model generates base probability  
- Rule-based adjustments applied for critical conditions:
  - Ghost touch issues  
  - Poor touch responsiveness  
  - Long maintenance gaps  
  - Corrective maintenance history  

---

### 🔹 4. Explainability
- Feature importance visualization  
- Supports:
  - Logistic Regression (coefficients)  
  - Tree-based models (feature importance)  
- Interactive Plotly charts  

---

### 🔹 5. Model Evaluation (Post-Mortem)
- Model performance summary  
- Stability insights  
- Limitations and real-world considerations  

---

## 📊 Technologies Used

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- Plotly  
- Matplotlib  

---

## 📁 Project Structure
├── dashboard/
│ └── app.py
├── models/
│ ├── model_registry.json
│ ├── model_history.json
│ └── trained_models.pkl
├── data/
│ └── datasets/
├── scripts/
│ ├── model_loader.py
│ ├── model_registry_builder.py
│ └── db_utils.py
└── README.md

## 🖥️ How to Run

```bash
python scripts/train_models.py
python -m streamlit run dashboard/app.py
