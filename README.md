# FactoryFlow AI

AI-powered manufacturing production forecasting system for industrial production lines.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/ML-Forecasting-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

FactoryFlow AI is a machine-learning-based forecasting platform designed to simulate and predict production quantities for manufacturing facilities.

The project focuses on:

- Production forecasting
- Machine-level analytics
- Operator efficiency modeling
- Feature engineering
- Manufacturing simulation
- Forecast model comparison

The simulation environment was designed around a cable manufacturing production facility.

---

## Features

- Daily & weekly production forecasting
- Machine / production line based prediction
- Synthetic manufacturing dataset generation
- Feature engineering pipeline
- Multiple model comparison
- Streamlit dashboard interface
- Forecast visualization
- Production analytics

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Streamlit
- Matplotlib

---

## Project Structure

```bash
factoryflow-ai/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Machine Learning Workflow

1. Data Simulation
2. Feature Engineering
3. Data Preprocessing
4. Model Training
5. Model Comparison
6. Forecast Evaluation
7. Streamlit Deployment

---

## Forecasting Features

Examples of engineered features:

- Product difficulty score
- Machine efficiency
- Operator performance
- Scrap rate
- Production duration
- Shift information
- Historical production averages
- Rolling statistics
- Lag features

---

## Model Comparison

The project includes comparison between multiple forecasting models to evaluate production prediction performance.

Example models:

- XGBoost
- Random Forest
- Linear Regression
- LightGBM

Performance evaluation metrics:

- RMSE
- MAE
- R² Score

---

## Dashboard Preview

> Streamlit dashboard screenshots can be added here.

```markdown
![Dashboard](images/dashboard.png)
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/factoryflow-ai.git
cd factoryflow-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

---

## Future Improvements

- Real-time production monitoring
- MES integration
- ERP integration
- Predictive maintenance module
- What-if production simulation
- API support

---

## Disclaimer

This project uses synthetic manufacturing data created for simulation and educational purposes.

---

## Author

Developed as an industrial AI forecasting and analytics project.
