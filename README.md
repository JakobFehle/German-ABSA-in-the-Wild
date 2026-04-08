# German Aspect-based Sentiment Analysis in the Wild: B2B Dataset Creation and Cross-Domain Evaluation

This repository contains the experimental setup, results, and processing scripts for our study on **Aspect-Based Sentiment Analysis (ABSA)** on German-language datasets, published at [KONVENS 2025](https://aclanthology.org/2025.konvens-1.19/).

---

## 🧠 Overview

Aspect-based sentiment analysis (ABSA) enables fine-grained sentiment extraction from user feedback by linking opinions to specific aspects mentioned in text. While significant progress has been made for English benchmarks, ABSA remains underexplored in many non-English languages and domain-specific contexts.

In this work, we:
- introduce a **multi-stage annotation study** on B2B software reviews,
- analyze challenges such as **implicit aspects** and **domain-specific phrasing**,
- and benchmark **state-of-the-art (SOTA) ABSA models** across multiple tasks and datasets.

We evaluate models on the following subtasks:
- **ACD** – Aspect Category Detection  
- **ACSA** – Aspect Category Sentiment Analysis  
- **TASD** – Target Aspect Sentiment Detection  

Our experiments span **five German datasets**, including a newly constructed B2B dataset.

---

## 🗂️ Repository Structure
```
📂 root/
├── 📂 data/                              # Datasets and processed data used for training and evaluation
│
├── 📂 results/                           # Experimental outputs for all evaluation runs
│   ├── 📂 <experiment_folder>/           # One folder per experiment configuration
│   │   ├── config.json                   # Model, dataset, and hyperparameter configuration
│   │   ├── predictions.json              # Model predictions and gold labels
│   │   ├── metrics_*.tsv                 # Evaluation results (e.g. F1, precision, recall)
│
├── 📂 scripts/                           # Scripts to run models and tasks
│   ├── experiment_script.py              # One script per model to run hyperparameter tuning and test set evaluation
│
├── 📂 src/                               # Core implementation of all ABSA methods
│   ├── 📂 bert_clf/                      # BERT-based classifiers (ACD, ACSA baselines)
│   ├── 📂 fs_llm/                        # Few-shot prompting with LLMs
│   ├── 📂 ft_llm/                        # Fine-tuned LLMs for structured ABSA tasks
│   ├── 📂 hier_gcn/                      # Graph-based ABSA (Hierarchical GCN)
│   ├── 📂 mvp/                           # Multi-View Prompting (MVP) implementation
│   ├── 📂 paraphrase/                    # Paraphrase-based TASD approaches
│   └── 📂 utils/                         # Shared helper functions (preprocessing, evaluation, etc.)
│
└── 📄 Annotation_Guidelines_English.pdf  # Guideliens for the annotation study
```

## 🧪 Experimental Setup

We evaluate multiple model types:

- **Classification models**
  - e.g. BERT-based architectures
- **Graph-based models**
  - e.g. Hierarchical GCN
- **Generative models**
  - Fine-tuned LLMs (e.g. instruction-tuned models)

---

## 📊 Datasets

### Dataset Overview

We use five German-language datasets:
- Existing domain datasets (e.g. MobASA, GermEval 2017, GERestaurant, Hotel Reviews)
- A newly constructed **B2B software feedback dataset**

| Name                     | Annotation     | Domain                        | Train | Dev  | Test | Total |
|--------------------------|---------------|--------------------------------|------:|-----:|-----:|------:|
| Hotel Reviews ([Fehle et al., 2023](https://aclanthology.org/2023.konvens-main.21/)) | AC, SP       | Hospitality (Hotels)          | 3403 |  –   | 851  | 4254 |
| MobASA ([Gabryszak et al., 2022](https://aclanthology.org/2022.csrnlp-1.5/))    | AT, AC, SP   | Public Transportation         | 3119 | 1054 | 1028 | 5201 |
| GERestaurant ([Hellwig et al., 2024](https://aclanthology.org/2024.konvens-main.14/))| AT, AC, SP   | Hospitality (Restaurants)     | 2135 |  –   | 919  | 3054 |
| GermEval ([Wojatzki et al., 2017](https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2017-wojatzkietal-germeval2017-workshop.pdf))   | AT, AC, SP   | Public Transportation         | 16200| 1917 | 3642 | 21759 |
| **B2B Software Reviews (Ours)**    | AT, AC, SP   | Software Products             | 1707 | 249  | 508  | 2464 |

**Abbreviations:**  
- AT = Aspect Term  
- AC = Aspect Category  
- SP = Sentiment Polarity  

⚠️ Note:  
The B2B dataset is **confidential**.  
We provide **synthetic examples** that reflect its structure and characteristics.

---

## 🔬 Results

### Aspect Category Detection (ACD)

| Method              | Hotel | MobASA | Rest | GermEval | B2B |
|---------------------|------:|------:|-----:|--------:|----:|
| BERT-CLF        | **89.06** | **94.07** | **91.09** | **78.10** | **75.60** |
| LLaMA Few-Shot      | 79.09 | 79.70 | 83.68 | 46.51 | 66.98 |
| LLaMA Fine-Tune     | 87.69 | 92.18 | 88.06 | 41.27 | 74.22 |

### Aspect Category Sentiment Analysis (ACSA)

| Method              | Hotel | MobASA | Rest | GermEval | B2B |
|---------------------|------:|------:|-----:|--------:|----:|
| BERT-CLF            | 78.75 | 83.57 | 84.34 | 65.83 | 69.71 |
| Hier-GCN            | 78.02 | 84.82 | 83.31 | **67.87** | **69.80** |
| LLaMA Few-Shot      | 74.80 | 70.29 | 80.79 | 39.36 | 64.78 |
| LLaMA Fine-Tune     | **80.51** | **87.22** | **85.22** | 33.41 | 69.13 |

### Target Aspect Sentiment Detection (TASD)

| Method              | MobASA | Rest | GermEval | B2B |
|---------------------|------:|-----:|--------:|----:|
| Paraphrase          | 78.69 | 65.72 | 54.03 | 50.27 |
| MvP                 | 79.65 | 67.00 | _55.75_ | 50.50 |
| LLaMA Few-Shot      | 64.62 | 61.13 | 43.78 | 42.34 |
| LLaMA Fine-Tune     | **81.56** | _73.22_ | 31.06 | **55.77** |

**Metric:** Micro-F1 (average over 5 seeds)  

## Key Findings

- **Simple classifiers** (e.g. BERT-based models) remain strong baselines for ACD.
- **Fine-tuned Large Language Models (LLMs)** perform best on structured tasks like ACSA and TASD.
- Model performance varies significantly across domains.
- Annotation quality and dataset design strongly influence results.
- ABSA methods do **not generalize uniformly** across domains.

---

## ✍️ Annotation Insights

Our multi-stage annotation process revealed key challenges:

- Handling **implicit aspects**
- Dealing with **domain-specific terminology**
- High variability in phrasing
- Importance of **clear annotation guidelines**
- Need for **iterative refinement and validation** during annotation

## 📌 Summary

This repository provides:
- A structured benchmark of ABSA methods in German
- A deep dive into B2B domain challenges
- Reproducible experiment outputs
- Tools for consistent label normalization

Our results highlight the importance of:
- **domain-aware modeling**
- **high-quality annotation**
- **task-specific evaluation strategies**

---

## 📬 Citation

```
@inproceedings{fehle-etal-2025-german,
    title = "{G}erman Aspect-based Sentiment Analysis in the Wild: {B}2{B} Dataset Creation and Cross-Domain Evaluation",
    author = "Fehle, Jakob  and
      Donhauser, Niklas  and
      Kruschwitz, Udo  and
      Hellwig, Nils Constantin  and
      Wolff, Christian",
    editor = "Wartena, Christian  and
      Heid, Ulrich",
    booktitle = "Proceedings of the 21st Conference on Natural Language Processing (KONVENS 2025): Long and Short Papers",
    month = sep,
    year = "2025",
    address = "Hannover, Germany",
    publisher = "HsH Applied Academics",
    url = "https://aclanthology.org/2025.konvens-1.19/",
    pages = "213--227"
}
```

---

Wenn du willst, kann ich dir noch eine **zweite Version für GitLab mit Badges + Paper-Link + BibTeX** machen (typisch für Paper-Repos).
