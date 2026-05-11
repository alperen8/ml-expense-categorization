# SEDS 537 – Machine Learning Individual Term Project
# Progress Report

**Title:** Uncertainty-Aware Expense Category Prediction for Household Finance Applications  
**Date:** May 11, 2026  

---

## 1. Project Title and Problem Definition

**Title:** Uncertainty-Aware Expense Category Prediction for Household Finance Applications

Personal finance applications rely on accurate expense categorization to generate meaningful budgets and spending summaries. In practice, transaction records contain short and sometimes ambiguous text descriptions such as merchant names, which makes automatic categorization non-trivial. The problem is further complicated by class imbalance, since common categories like Groceries dominate while others like Haircut appear rarely.

This project formulates expense categorization as a **multi-class text classification** task. The input to the system is a transaction record consisting of a merchant description (e.g., "Shell", "Grocery Store"), the transaction amount, the transaction date, and the account type. The output is one of 16 expense categories such as Gas & Fuel, Restaurants, or Utilities. The core challenge is that (1) many merchant names are ambiguous, (2) the same merchant can appear in different categories depending on context, and (3) the dataset is moderately imbalanced with some categories having very few examples. A secondary goal is to add confidence estimation so that uncertain predictions can be flagged rather than silently accepted.

---

## 2. Dataset and Preprocessing Status

**Dataset:** Personal Transactions (personal_transactions.csv)  
**Source:** Kaggle (publicly available personal finance EDA notebook inputs)  
**Status:** Downloaded, cleaned, and prepared for experiments ✓

### Dataset Summary

| Property | Value |
|---|---|
| Total records (raw) | 808 |
| Records after filtering | 613 |
| Number of categories | 16 |
| Date range | January 2018 – September 2019 |
| Input features | Description (text), Amount, Date, Transaction Type, Account Name |
| Target label | Category |

### Preprocessing Steps Completed

1. **Filtered non-expense rows:** Removed "Credit Card Payment" and "Paycheck" entries, which are transfers and income rather than expense categories.
2. **Category merging:** Merged semantically equivalent low-frequency categories: "Television" → "Movies & DVDs", "Food & Dining" → "Restaurants", "Entertainment" → "Movies & DVDs".
3. **Rare class removal:** Dropped categories with fewer than 5 samples to ensure stratified splitting is possible.
4. **Feature engineering:** Extracted month and weekday from the date; computed log-transformed amount; encoded transaction type (debit/credit) as a binary flag.
5. **Train/validation/test split:** 70% / 15% / 15%, stratified by category.

| Split | Samples |
|---|---|
| Train | 429 |
| Validation | 92 |
| Test | 92 |

### Known Challenges

- **Class imbalance:** Groceries has 105 samples while Haircut has only 13. This causes poor recall on minority classes.
- **Ambiguous descriptions:** Generic names such as "Grocery Store" or "Hardware Store" appear repeatedly, while specific names like "Chick-Fil-A" or "Starbucks" are easily recognizable.
- **Small dataset:** 613 samples limits the use of deep learning models. Lightweight models are more appropriate.

---

## 3. Literature Review Progress

The following methods and papers have been reviewed so far:

**TF-IDF based text classification:** TF-IDF remains a strong baseline for short-text classification tasks. Joulin et al. (2017, FastText) showed that simple linear models over bag-of-words or n-gram features often match deep learning on short-text classification. This motivates the TF-IDF baselines in this project.

**Transaction categorization literature:** Existing work on bank transaction categorization (e.g., Xue et al., 2022; Srinivasan et al., 2020) generally uses a combination of merchant name, transaction amount, and temporal features. These papers confirm that merchant name is the dominant signal and that structured features add marginal but consistent improvement.

**Sentence embeddings:** Reimers and Gurevych (2019, Sentence-BERT) introduced efficient sentence encoders that can represent transaction descriptions as dense vectors. This is the planned direction for the proposed method.

**Calibration:** Guo et al. (2017, "On Calibration of Modern Neural Networks") provides the theoretical basis for temperature scaling, which will be used in the confidence estimation component.

**Key direction:** The reviewed literature consistently shows that text is the dominant feature for this type of classification, and that lightweight calibrated classifiers outperform uncalibrated deep models on small datasets.

---

## 4. Baseline Models

Three baselines were selected and fully implemented:

| Baseline | Justification | Status |
|---|---|---|
| Majority Class | Lower bound reference; predicts the most frequent class for all inputs | Implemented ✓ |
| TF-IDF + Logistic Regression | Standard text classification baseline; well-calibrated probabilities via softmax | Implemented ✓ |
| TF-IDF + Linear SVM | Strong discriminative baseline for text; commonly used in short-text tasks | Implemented ✓ |

All baselines use unigram + bigram TF-IDF features on the Description column, combined with structured features (log-amount, month, weekday, is_debit). Scripts are in `src/baselines.py`.

---

## 5. Initial Experimental Results

All three baselines were evaluated on the held-out test set (92 samples). Results are reported below:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Majority Class | 0.1739 | 0.0185 | 0.0515 |
| TF-IDF + Logistic Regression | **0.9565** | **0.9160** | **0.9477** |
| TF-IDF + Linear SVM | 0.9457 | 0.9111 | 0.9410 |

**Interpretation:**

- The majority class baseline confirms the expected lower bound: always predicting "Groceries" yields 17% accuracy, which demonstrates meaningful class imbalance.
- TF-IDF + Logistic Regression achieves 95.65% accuracy and 0.916 macro F1, indicating that merchant names are highly informative and largely unambiguous in this dataset.
- TF-IDF + Linear SVM performs similarly (94.57%), slightly below LR, which is consistent with LR having better probability estimates.
- Both text models fail on "Fast Food" in the test set (0 F1), likely because only 2 test samples exist and some fast food merchants (e.g., "Bojangles") overlap with restaurants in description style.
- The strong baseline performance raises an interesting challenge for the proposed method: the marginal gains from a more complex model will be small on clean merchant names, but the uncertainty estimation component becomes the primary differentiator. The focus will shift to identifying which predictions are likely to be wrong.

---

## 6. Planned Improvements and Technical Direction

Given the strong baseline performance, the next phase of the project will focus on two directions:

**6.1 Proposed Method: Sentence Embedding + Calibrated Classifier**

The proposed method will replace TF-IDF with a pretrained sentence encoder (e.g., all-MiniLM-L6-v2 from Sentence-Transformers) to produce dense 384-dimensional embeddings of transaction descriptions. These embeddings will be concatenated with structured features and passed to a logistic regression or shallow MLP head. Confidence calibration using temperature scaling will then be applied on the validation set.

**6.2 Handling Class Imbalance**

Class-weighted loss functions and oversampling of minority categories will be tested.

**6.3 Ablation Variants**

- Text only (no structured features)
- Structured features only (no text)
- Text + structured (current setup)
- With and without confidence calibration

---

## 7. Ablation and Error Analysis Plan

**Ablation Study:**

The ablation will compare four feature configurations on the same model (Logistic Regression):
1. Description TF-IDF only
2. Structured features only (amount, date, account type)
3. TF-IDF + structured features (current)
4. Sentence embedding + structured features (proposed)

This will quantify the contribution of each input modality.

**Error Analysis Plan:**

- Examine all misclassified test samples and group them by true category
- Identify which category pairs are most frequently confused (e.g., Restaurants vs. Fast Food)
- Analyze whether misclassified samples have low model confidence
- Look at edge cases: generic merchant names (e.g., "Brunch Restaurant") and minority classes

---

## 8. Visualization and Interpretation

The following visualizations are planned or partially implemented:

| Visualization | Status |
|---|---|
| Category class distribution bar chart | Done ✓ |
| Confusion matrix (LR) | Done ✓ |
| Confusion matrix (SVM) | Done ✓ |
| Learning curves | Planned |
| Calibration plot (reliability diagram) | Planned |
| t-SNE of sentence embeddings | Planned |
| Per-class F1 comparison across models | Planned |

All saved figures are in the `results/` directory.

---

## 9. GitHub and Reproducibility Status

**Repository:** To be created and made public this week.

**Current Local Repository Structure:**

```
ml-expense-categorization/
├── data/
│   └── personal_transactions.csv
├── src/
│   └── baselines.py
├── results/
│   ├── baseline_results.csv
│   ├── class_distribution.png
│   ├── cm_tf-idf_+_logistic_regression.png
│   └── cm_tf-idf_+_linear_svm.png
├── requirements.txt
└── README.md
```

**Implemented Scripts:**
- `src/baselines.py` — full pipeline: loading, preprocessing, feature extraction, three baseline models, evaluation, and visualization

**Dependencies:** Listed in `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, seaborn, scipy)

**Running the Experiments:**
```bash
pip install -r requirements.txt
python src/baselines.py
```

**Reproducibility:** Fixed random seed (42) is used in all train/test splits and model initializations.

---

## 10. Current Challenges and Next Steps

**Current Challenges:**

- **Dataset size:** 613 samples is relatively small for deep learning approaches. Sentence-BERT embeddings will be evaluated but may overfit without regularization.
- **Minority class performance:** Categories with fewer than 15 samples (e.g., Haircut, Fast Food) show poor recall. Class-weighting and oversampling will be explored.
- **Near-ceiling baseline:** The text baselines already achieve ~95% accuracy, making it difficult to show substantial improvement. The uncertainty estimation component will be the main contribution.

**Next Steps Before Final Submission:**

1. Push project to public GitHub repository
2. Implement sentence embedding model (proposed method)
3. Apply temperature scaling for confidence calibration
4. Run ablation study across four feature configurations
5. Conduct systematic error analysis on misclassified samples
6. Generate calibration plots and t-SNE visualizations
7. Write final report with complete results and analysis

---

*Submitted for SEDS 537 – Machine Learning Individual Term Project Progress Report*  
*Submission Deadline: May 11, 2026*
