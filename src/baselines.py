"""
SEDS 537 – ML Term Project
Expense Category Prediction: Baseline Models
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "personal_transactions.csv")

EXCLUDED_CATEGORIES = {"Credit Card Payment", "Paycheck"}

CATEGORY_MAP = {
    "Television": "Movies & DVDs",
    "Food & Dining": "Restaurants",
    "Entertainment": "Movies & DVDs",
}

MIN_SAMPLES = 5


def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df[~df["Category"].isin(EXCLUDED_CATEGORIES)].copy()
    df = df.dropna(subset=["Description", "Category", "Amount"])

    df["Category"] = df["Category"].replace(CATEGORY_MAP)

    counts = df["Category"].value_counts()
    valid_cats = counts[counts >= MIN_SAMPLES].index
    df = df[df["Category"].isin(valid_cats)].copy()

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["month"] = df["Date"].dt.month
    df["weekday"] = df["Date"].dt.weekday
    df["is_debit"] = (df["Transaction Type"] == "debit").astype(int)
    df["log_amount"] = np.log1p(df["Amount"])

    return df


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["Category"], random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["Category"], random_state=random_state
    )
    return train, val, test


def build_features(train, val, test):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    X_train_text = tfidf.fit_transform(train["Description"])
    X_val_text = tfidf.transform(val["Description"])
    X_test_text = tfidf.transform(test["Description"])

    meta_cols = ["log_amount", "month", "weekday", "is_debit"]
    X_train_meta = csr_matrix(train[meta_cols].values)
    X_val_meta = csr_matrix(val[meta_cols].values)
    X_test_meta = csr_matrix(test[meta_cols].values)

    X_train = hstack([X_train_text, X_train_meta])
    X_val = hstack([X_val_text, X_val_meta])
    X_test = hstack([X_test_text, X_test_meta])

    return X_train, X_val, X_test, tfidf


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"Model": name, "Accuracy": round(acc, 4), "Macro F1": round(macro_f1, 4), "Weighted F1": round(weighted_f1, 4)}


def plot_confusion_matrix(name, model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {fname}")


def plot_class_distribution(df):
    counts = df["Category"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Category Distribution (after filtering)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Class distribution saved: {fname}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)

    print(f"\nDataset after filtering: {len(df)} samples, {df['Category'].nunique()} categories")
    print("\nClass counts:")
    print(df["Category"].value_counts().to_string())

    plot_class_distribution(df)

    train, val, test = split_data(df)
    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    y_train = train["Category"].values
    y_val = val["Category"].values
    y_test = test["Category"].values
    labels = sorted(df["Category"].unique())

    X_train, X_val, X_test, tfidf = build_features(train, val, test)

    results = []

    # Baseline 1: Majority class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    results.append(evaluate("Majority Class", dummy, X_test, y_test))

    # Baseline 2: TF-IDF + Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    results.append(evaluate("TF-IDF + Logistic Regression", lr, X_test, y_test))
    plot_confusion_matrix("TF-IDF + Logistic Regression", lr, X_test, y_test, labels)

    # Baseline 3: TF-IDF + Linear SVM
    svm = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    results.append(evaluate("TF-IDF + Linear SVM", svm, X_test, y_test))
    plot_confusion_matrix("TF-IDF + Linear SVM", svm, X_test, y_test, labels)

    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    print(f"\nResults saved to results/baseline_results.csv")


if __name__ == "__main__":
    main()
