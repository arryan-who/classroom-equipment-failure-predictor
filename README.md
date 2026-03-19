# Classroom Equipment Failure Predictor

## 📌 Overview
This project is a machine learning-based system designed to monitor classroom equipment and estimate their health status. It helps in proactive maintenance by analyzing operational conditions and providing actionable insights.

---

## 🎯 Objective
- Predict equipment health using data-driven methods  
- Reduce unexpected failures  
- Enable proactive maintenance decisions  

---

## 🧠 Key Features
- Equipment-specific machine learning models  
- Real-time health analysis via dashboard  
- Interactive simulation of operating conditions  
- Clear maintenance recommendations  

---

## 🏗️ System Architecture
- Synthetic Data Generation  
- Model Training (Random Forest)  
- Dashboard (Streamlit Interface)  

---

## ⚙️ Equipment Covered
- Projector  
- Smartboard  
- Lighting  
- Air Conditioner  

Each equipment has a separate model due to different behavior patterns.

---

## 📊 Features Used
- Equipment age  
- Usage hours  
- Maintenance gap  
- Power fluctuations  
- Equipment-specific parameters  

---

## 🤖 Machine Learning Model
- Model: Random Forest Classifier  
- Type: Supervised Learning  
- Output: Probability converted into system health status  

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

## 🧪 Data Source
Synthetic dataset generated using:
- Risk scoring logic  
- Sigmoid probability  
- Random noise for realism  

---

## ⚠️ Limitations
- Uses synthetic data  
- Requires real-world validation  

---

## 🚀 Future Scope
- IoT sensor integration  
- Cloud deployment  
- Automated maintenance scheduling  

---

## 🖥️ How to Run

```bash
python scripts/train_models.py
python -m streamlit run dashboard/app.py
