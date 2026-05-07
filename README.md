# Expected Credit Loss (CECL) Modeling

End-to-end credit risk framework estimating **Current Expected Credit Loss (CECL)** through component-wise modeling of Probability of Default (PD), Exposure at Default (EAD), and Loss Given Default (LGD).

---

## 🎯 Business Problem

Financial institutions under CECL standards (ASC 326) must estimate lifetime expected losses on loan portfolios — not just incurred losses. This project builds a modular framework that mirrors how banks and credit risk teams approach this problem in practice.

---

## 🧠 Modeling Framework

CECL is computed as:

```
CECL = PD × EAD × LGD
```

Each component is modeled independently:

### 1. Probability of Default (PD)
- **Logistic regression** model trained on historical loan performance features
- Features include delinquency history, credit utilization, loan age, and payment behavior
- Outputs a probability score per account representing likelihood of default over the forecast horizon

### 2. Exposure at Default (EAD)
- Calculated as the ratio of unpaid principal to original open balance
- Accounts for loan amortization and partial paydowns over the forecast period

### 3. Loss Given Default (LGD)
- Estimated as the expected loss rate conditional on default occurring
- Derived from historical recovery rates on defaulted accounts

---

## 📊 Pipeline

```
Loan Performance Data
        │
        ▼
Data Cleaning & Risk State Assignment
        │
        ├──▶ PD: Transition Matrix Estimation → Multi-period PD
        ├──▶ EAD: Unpaid / Open Balance Ratio
        └──▶ LGD: Historical Recovery Rate Estimation
                │
                ▼
        CECL = PD × EAD × LGD (per account)
                │
                ▼
        Portfolio Aggregation & Validation vs. Actuals
```

---

## 📈 Evaluation

- Predicted CECL is compared against actual realized losses
- Model performance assessed on portfolio-level and segment-level accuracy

---

## 🛠 Stack

- **pandas, NumPy** — data processing and matrix operations
- **scikit-learn** — model development and evaluation
- **matplotlib, seaborn** — risk state visualization and loss distribution plots

---

## 📁 Structure

```
├── Expected_Credit_Loss.ipynb    # Full modeling notebook
├── data/                         # Loan performance dataset
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Expected_Credit_Loss.ipynb
```

---

## 📌 Related Projects

- [Customer Lifetime Value](https://github.com/Ajay-Deshpande/Customer-Lifetime-Value) — probabilistic modeling for customer value
- [Scalable Time Series Forecasting](https://github.com/Ajay-Deshpande/Scalable-Time-Series-Forecasting) — forecasting framework
