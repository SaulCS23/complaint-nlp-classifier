# Consumer Complaint NLP Classifier

A reproducible **TF-IDF + Linear SVM** pipeline for classifying consumer-finance complaints. This portfolio implementation is inspired by my research on the CFPB Consumer Complaint Database. Previous research results are distinct from this demonstration.

## Problem

Route incoming complaints into four categories: `credit_reporting`, `debt_not_owed`, `collection_practices`, and `debt_verification`.

## Previous research (not reproduced by the sample)

My earlier research processed millions of CFPB complaint records, mapped 173 issue labels into four groups, and used a balanced subset of 200,000 complaints (160,000 training / 40,000 test). A tuned Linear SVM achieved **71.94% held-out accuracy**. These are historical research findings; the synthetic sample included here does **not** reproduce those metrics.

## Architecture

```text
CSV complaints -> validate/filter -> stratified train/test split
       -> TF-IDF (unigrams+bigrams, <=20k features)
       -> LinearSVC -> classification report + confusion matrix
                    -> serialized model -> predict new complaint
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m complaint_classifier.train --data data/sample_complaints.csv
python -m complaint_classifier.predict --text 'My credit report lists an account I never opened'
pytest -q
```

The included 24-row synthetic CSV is a **smoke test only**, not a valid benchmark. To evaluate real-world performance, download the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/), map the original issue labels to the four categories, and provide a CSV with `text` and `label` columns. The original issue-to-category mapping and preprocessing decisions must be documented to claim faithful replication.

## Engineering choices

- Stratified 80/20 split with fixed seed
- Pipeline prevents training/test TF-IDF leakage
- Saved metrics include per-class precision, recall, F1, and confusion matrix
- Unit tests cover training artifacts and input validation
- No private company data or production credentials

## Limitations

The synthetic demo is too small to establish performance. Full replication requires the original label mapping, deduplication, temporal holdout, error analysis, and reproducible dataset provenance. Do not use this model to make automated financial decisions.

## Author

Saul Assis | [GitHub](https://github.com/SaulCS23)
