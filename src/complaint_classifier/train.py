"""Train and evaluate a TF-IDF + LinearSVC complaint classifier."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def train(csv_path, output_dir, text_column="text", label_column="label", test_size=0.2, seed=42):
    df = pd.read_csv(csv_path)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[[text_column, label_column]].dropna()
    df = df[(df[text_column].astype(str).str.strip() != "") & (df[label_column].astype(str).str.strip() != "")]
    if len(df) < 10 or df[label_column].nunique() < 2:
        raise ValueError("Need at least 10 nonempty records and two classes")
    counts = df[label_column].value_counts()
    if counts.min() < 2:
        raise ValueError("Each class needs at least two examples for stratified splitting")
    x_train, x_test, y_train, y_test = train_test_split(
        df[text_column].astype(str), df[label_column].astype(str),
        test_size=test_size, random_state=seed, stratify=df[label_column]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
        ("svm", LinearSVC(random_state=seed, max_iter=5000)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(df[label_column].astype(str).unique())
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "train_samples": len(x_train), "test_samples": len(x_test),
        "labels": labels,
        "classification_report": classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV containing text and label columns")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()
    metrics = train(args.data, args.output, args.text_column, args.label_column)
    print(f"Accuracy: {metrics['accuracy']:.3f} on {metrics['test_samples']} held-out examples")


if __name__ == "__main__":
    main()
