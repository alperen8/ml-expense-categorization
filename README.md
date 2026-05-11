# Uncertainty-Aware Expense Category Prediction

SEDS 537 – Machine Learning Individual Term Project

## Problem
Predicting expense categories from transaction descriptions and metadata using supervised ML, with confidence estimation for low-certainty predictions.

## Dataset
`personal_transactions.csv` — personal finance transaction records (2018–2019)  
Columns: `Date`, `Description`, `Amount`, `Transaction Type`, `Category`, `Account Name`

Place the CSV file in the `data/` folder before running.

## Setup

```bash
pip install -r requirements.txt
```

## Run Baselines

```bash
python src/baselines.py
```

Results are saved to `results/baseline_results.csv`.

## Project Structure

```
ml-expense-categorization/
├── data/
│   └── personal_transactions.csv
├── src/
│   └── baselines.py
├── results/
│   └── baseline_results.csv
├── requirements.txt
└── README.md
```
