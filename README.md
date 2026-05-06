# 🏦 IFRS 9 Expected Credit Loss (ECL) AI Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-XGBoost_%7C_Isolation_Forest-orange)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-success)

## 📌 Executive Summary
This project is an enterprise-grade, end-to-end quantitative finance pipeline designed to calculate Expected Credit Loss (ECL) provisions in strict accordance with the **IFRS 9 Regulatory Standard**. 

Unlike traditional static actuarial models, this engine integrates **Machine Learning** to dynamically predict Probability of Default (PD) and detect behavioral anomalies, merging advanced data science with strict banking regulations. 

The project encompasses a full data engineering pipeline, machine learning modeling, an actuarial math engine, automated testing, and an interactive executive web dashboard.

## 🤝 Collaboration
This enterprise quantitative risk engine was architected and developed in collaboration with **Masoma Shai**.

---

## 📸 Executive Dashboard & Auditing

### IFRS 9 Portfolio Risk Dashboard
![Streamlit Dashboard](assets/executive_dashboard.jpg)
*Interactive UI built with Streamlit allowing risk managers to filter the portfolio by IFRS 9 Stage, view total Exposure at Default (EAD), and sort high-risk accounts.*

### SHAP Explainability Auditor Report
![SHAP Auditor Report](assets/shap_auditor_report.png)
*AI Model auditing using SHAP values to explain the driving factors behind Probability of Default predictions, ensuring regulatory transparency.*

---

## ⚙️ System Architecture & Pipeline

The system is built sequentially across 5 distinct phases:

1. **Synthetic Data Generation (`data_generator`)**
   * Generates a realistic, synthetic retail/SME banking portfolio.
   * Creates core variables including FICO equivalents, Days Past Due (DPD) buckets, Loan-to-Income ratios, and outstanding balances.
2. **AI Behavioral Anomaly Detection (`ml_pipeline`)**
   * Utilizes **Isolation Forests** to scan debtor behavioral data for hidden risk anomalies prior to outright default.
3. **AI Probability of Default Modeling (`ml_pipeline`)**
   * Utilizes an ML classifier to calculate a granular, forward-looking Probability of Default (PD) for every individual loan.
4. **IFRS 9 Actuarial Engine (`actuarial_engine`)**
   * Translates IFRS 9 legislation into strict Python logic.
   * **Stage 1:** Performing loans (12-month ECL).
   * **Stage 2:** Significant Increase in Credit Risk (SICR) triggered by PD multiples > 2.0x, 30+ DPD (without rebuttable presumption), or forbearance flags (Lifetime ECL).
   * **Stage 3:** Credit Impaired / 90+ DPD (Lifetime ECL).
   * Calculates the final Expected Credit Loss (ZAR) provision.
5. **Interactive Executive Reporting (`dashboard`)**
   * A Streamlit web application that queries the SQLite database in real-time to visualize provisions, high-risk accounts, and portfolio distribution.

---

## 🛠️ Technology Stack

* **Languages:** Python 3.10+
* **Data & Math:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Isolation Forest), XGBoost, SHAP (for model explainability)
* **Database:** SQLite3 (`ifrs9_engine.db`)
* **Frontend:** Streamlit
* **DevOps & Testing:** Pytest, GitHub Actions (CI/CD)

---

## 📂 Project Structure

```text
IFRS9-ECL-ENGINE/
│
├── src/
│   ├── data_generator/      # Synthetic banking data creation
│   ├── ml_pipeline/         # AI models (Anomaly Detection & PD Models)
│   ├── actuarial_engine/    # IFRS 9 Staging logic and ECL math
│   └── dashboard/           # Streamlit interactive UI
│
├── tests/                   # Pytest suite (Math verification & Database checks)
├── outputs/                 # Stores the generated ifrs9_engine.db SQLite database
├── assets/                  # Dashboard screenshots and SHAP reports
└── README.md
