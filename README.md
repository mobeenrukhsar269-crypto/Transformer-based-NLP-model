# NLP Assignment 4 — Transformer-Based Sentiment Classification with Explainability

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-lightgrey)

---

## Overview

This project fine-tunes a **DistilBERT** transformer model for binary sentiment classification on the **Amazon Polarity** dataset. Beyond training and evaluation, the project applies two explainability techniques — **SHAP** and **LIME** — to interpret model predictions and compares their runtime performance.

---

## Dataset

| Property | Details |
|---|---|
| Dataset | [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) |
| Task | Binary Sentiment Classification (Positive / Negative) |
| Training Samples | 16,000 (subset) |
| Validation Samples | 4,000 (subset) |
| Test Samples | 5,000 (subset) |

---

## Model

| Property | Details |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Task Head | Sequence Classification (2 labels) |
| Max Token Length | 128 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 2 |
| Weight Decay | 0.01 |
| Optimizer | AdamW (default HuggingFace Trainer) |

---

## Project Structure

```
TRANSFORMER_NLP_PROJECT/
│
├── F230048_NLP_ASS4.ipynb      # Main notebook (all code and outputs)
├── README.md                   # Project documentation
│
├── data/                       # Dataset (loaded via HuggingFace datasets)
├── models/                     # Saved model weights (amazon_sentiment_model/)
└── outputs/
    ├── lime/                   # LIME explanation HTML files
    └── shap/                   # SHAP explanation outputs
```

---

## Pipeline

```
Load Dataset (Amazon Polarity)
        │
        ▼
Tokenize with DistilBERT Tokenizer
        │
        ▼
Fine-tune DistilBERT (HuggingFace Trainer)
        │
        ▼
Evaluate (Accuracy, F1, Precision, Recall, Confusion Matrix)
        │
        ▼
Attention Weight Visualization
        │
        ▼
SHAP Explainability (20 samples)
        │
        ▼
LIME Explainability (20 samples)
        │
        ▼
Runtime Comparison (SHAP vs LIME)
        │
        ▼
Misclassification Analysis
```

---

## Explainability Methods

### SHAP (SHapley Additive exPlanations)
- Uses `shap.Explainer` with a HuggingFace `text-classification` pipeline
- Token-level attribution scores visualized using `shap.plots.text`
- Applied to 20 test samples

### LIME (Local Interpretable Model-agnostic Explanations)
- Uses `LimeTextExplainer` with a custom `predict_proba` wrapper
- Top 10 features highlighted per explanation
- Applied to 20 test samples
- Outputs saved as HTML files to Google Drive

### Runtime Comparison
- Single-sample inference time measured for both SHAP and LIME
- Results visualized as a bar chart

---

## Results

| Metric | Value |
|---|---|
| Accuracy | See notebook output |
| F1 Score | See notebook output |
| Precision | See notebook output |
| Recall | See notebook output |

> Full classification report and confusion matrix are available in the notebook.

---

## Requirements

```
torch
transformers
datasets
accelerate
evaluate
shap
lime
seaborn
matplotlib
scikit-learn
numpy
pandas
```

Install all dependencies:

```bash
pip install transformers datasets accelerate evaluate shap lime seaborn -q
```

---

## How to Run

1. Open `F230048_NLP_ASS4.ipynb` in **Google Colab**
2. Set runtime to **GPU** (T4 or better): `Runtime → Change runtime type → GPU`
3. Mount Google Drive when prompted (for saving outputs)
4. Run all cells sequentially

> **Note:** The LIME cells require `torch.no_grad()` and `num_samples=500` to avoid CUDA out-of-memory errors on a T4 GPU.

---

## Author

**Roll No:** F230048  
**Course:** Natural Language Processing  
**Submitted to:** [Your Instructor Name]  
**Institution:** [Your University Name]