from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "SEDS 537 - Machine Learning Individual Term Project | Progress Report", align="C")
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 7, title, fill=True, ln=True)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(220, 220, 220)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_fill_color(250, 250, 250)
        for j, row in enumerate(rows):
            fill = j % 2 == 0
            self.set_fill_color(248, 248, 248) if fill else self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill)
            self.ln()
        self.ln(3)


pdf = PDF()
pdf.set_margins(15, 15, 15)
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title block
pdf.set_font("Helvetica", "B", 14)
pdf.set_text_color(20, 20, 20)
pdf.cell(0, 8, "Uncertainty-Aware Expense Category Prediction", ln=True, align="C")
pdf.cell(0, 8, "for Household Finance Applications", ln=True, align="C")
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 6, "SEDS 537 | Progress Report | May 11, 2026", ln=True, align="C")
pdf.ln(4)
pdf.set_draw_color(80, 80, 80)
pdf.line(15, pdf.get_y(), 195, pdf.get_y())
pdf.ln(5)

# 1. Problem Definition
pdf.section_title("1. Project Title and Problem Definition")
pdf.body(
    "This project addresses the automatic categorization of personal finance transactions using machine "
    "learning. Given a transaction record consisting of a merchant description (e.g., 'Shell', 'Grocery Store'), "
    "transaction amount, and date, the system predicts one of 16 expense categories such as Gas & Fuel, "
    "Restaurants, or Utilities. The ML task is multi-class text classification.\n\n"
    "The problem is motivated by real-world household finance applications where accurate categorization "
    "drives budgeting, analytics, and spending summaries. Challenges include class imbalance, ambiguous "
    "merchant names, and noisy short-text descriptions. A secondary goal is uncertainty estimation: rather "
    "than silently accepting all predictions, the system will flag low-confidence outputs for user review.\n\n"
    "This work is also motivated by a broader graduation project on mobile expense tracking, where transaction "
    "data is captured via OCR from physical receipts. In future iterations, the ML pipeline developed here "
    "will be extended to handle Turkish-language transaction descriptions from OCR output, making cross-lingual "
    "generalization a natural next step."
)

# 2. Dataset
pdf.section_title("2. Dataset and Preprocessing Status")
pdf.body("Dataset: personal_transactions.csv | Source: Kaggle (public personal finance EDA notebooks)\nStatus: Downloaded, cleaned, and prepared for experiments.")

pdf.table(
    ["Property", "Value"],
    [
        ["Raw records", "808"],
        ["After filtering", "613"],
        ["Categories", "16"],
        ["Date range", "Jan 2018 - Sep 2019"],
        ["Train / Val / Test", "429 / 92 / 92 (stratified)"],
    ],
    [90, 100]
)

pdf.body(
    "Preprocessing steps: (1) Removed non-expense rows (Credit Card Payment, Paycheck). "
    "(2) Merged low-frequency semantically equivalent categories (Television -> Movies & DVDs, "
    "Food & Dining -> Restaurants). (3) Dropped categories with fewer than 5 samples. "
    "(4) Extracted month, weekday, and log-amount as structured features. "
    "Known challenges include class imbalance (Groceries: 105 samples vs. Haircut: 13) "
    "and generic merchant names that may map to multiple categories."
)

# 3. Literature Review
pdf.section_title("3. Literature Review Progress")
pdf.body(
    "The following works have been reviewed:\n"
    "- Joulin et al. (2017, FastText): Demonstrated that linear models over n-gram TF-IDF features "
    "match deep learning on short-text classification, motivating the TF-IDF baselines.\n"
    "- Reimers & Gurevych (2019, Sentence-BERT): Introduced efficient dense sentence embeddings "
    "that capture semantic similarity beyond surface n-grams. Planned for the proposed method.\n"
    "- Guo et al. (2017, 'On Calibration of Modern Neural Networks'): Provides the theoretical basis "
    "for temperature scaling, which will be used for confidence calibration.\n"
    "- Transaction categorization literature (Xue et al. 2022; Srinivasan et al. 2020): Confirms that "
    "merchant name is the dominant signal, and that structured features (amount, date) add marginal gains.\n\n"
    "Overall direction: lightweight calibrated classifiers over text + structured features, with a "
    "pretrained sentence encoder as the proposed improvement over TF-IDF baselines."
)

# 4. Baselines
pdf.section_title("4. Baseline Models")
pdf.table(
    ["Model", "Justification", "Status"],
    [
        ["Majority Class", "Lower bound; predicts most frequent class", "Implemented"],
        ["TF-IDF + Logistic Regression", "Standard text baseline; calibrated probabilities", "Implemented"],
        ["TF-IDF + Linear SVM", "Discriminative text baseline; strong on short text", "Implemented"],
    ],
    [55, 95, 40]
)
pdf.body(
    "All baselines use unigram + bigram TF-IDF on the Description field combined with structured "
    "features (log-amount, month, weekday, is_debit). Scripts are in src/baselines.py."
)

# 5. Results
pdf.section_title("5. Initial Experimental Results")
pdf.table(
    ["Model", "Accuracy", "Macro F1", "Weighted F1"],
    [
        ["Majority Class", "0.1739", "0.0185", "0.0515"],
        ["TF-IDF + Logistic Regression", "0.9565", "0.9160", "0.9477"],
        ["TF-IDF + Linear SVM", "0.9457", "0.9111", "0.9410"],
    ],
    [80, 37, 37, 36]
)
pdf.body(
    "The majority class baseline confirms meaningful class imbalance: predicting 'Groceries' every time "
    "yields only 17% accuracy. TF-IDF + Logistic Regression achieves 95.65% accuracy and 0.916 macro F1, "
    "showing that merchant descriptions are highly informative. Linear SVM performs comparably (94.57%). "
    "Both text models fail on 'Fast Food' in the test set (0 F1) due to very few test samples (n=2) "
    "and ambiguous merchant names that overlap with Restaurants. "
    "The high baseline performance shifts the focus of the proposed method toward uncertainty estimation "
    "rather than raw accuracy improvement."
)

# 6 & 7 on second page
pdf.section_title("6. Planned Improvements and Technical Direction")
pdf.body(
    "The proposed method will replace TF-IDF with a pretrained sentence encoder (all-MiniLM-L6-v2, "
    "Sentence-Transformers) producing 384-dimensional dense embeddings. These will be concatenated with "
    "structured features and passed to a logistic regression head. Temperature scaling will be applied "
    "on the validation set for confidence calibration, producing an Expected Calibration Error (ECE) metric.\n\n"
    "Additional directions: class-weighted loss for minority categories, hyperparameter tuning via grid search, "
    "and ablation across four feature configurations (text only, structured only, text+structured, "
    "embedding+structured)."
)

pdf.section_title("7. Ablation and Error Analysis Plan")
pdf.body(
    "Ablation study: Four feature configurations will be compared using Logistic Regression as the fixed "
    "classifier, isolating the contribution of text vs. structured features and TF-IDF vs. embeddings.\n\n"
    "Error analysis: All misclassified test samples will be examined by category pair. Focus areas include "
    "Restaurants vs. Fast Food confusion, generic merchant names (e.g., 'Hardware Store'), minority class "
    "failures, and overconfident wrong predictions as revealed by the calibration analysis."
)

pdf.section_title("8. Visualization Plan")
pdf.body(
    "Completed: class distribution bar chart, confusion matrices for LR and SVM (saved in results/).\n"
    "Planned: learning curves, calibration reliability diagram, t-SNE of sentence embeddings "
    "colored by category, per-class F1 bar chart comparing all models."
)

pdf.section_title("9. GitHub and Reproducibility Status")
pdf.body(
    "Repository: https://github.com/alperen8/ml-expense-categorization (public)\n\n"
    "Repository structure:\n"
    "ml-expense-categorization/ -> data/, src/baselines.py, results/, requirements.txt, README.md\n\n"
    "Running the experiments: pip install -r requirements.txt && python src/baselines.py\n"
    "Fixed random seed (42) is used in all splits and model initializations. "
    "Dataset preparation instructions are in README.md. "
    "Commit history reflects consistent development from project setup through baseline implementation."
)

pdf.section_title("10. Current Challenges and Next Steps")
pdf.body(
    "Challenges: (1) Small dataset (613 samples) limits deep learning; sentence embeddings may overfit "
    "without regularization. (2) Near-ceiling text baselines (~95%) make raw accuracy improvement difficult; "
    "the uncertainty component becomes the primary contribution. (3) Minority classes (Haircut, Fast Food) "
    "remain difficult.\n\n"
    "Next steps: implement sentence embedding model, apply temperature scaling, run ablation study, "
    "complete error analysis, generate calibration plots, push to GitHub, and write the final report."
)

out_path = "/Users/alperenertan/Desktop/ml-expense-categorization/progress_report.pdf"
pdf.output(out_path)
print(f"PDF saved: {out_path}")
