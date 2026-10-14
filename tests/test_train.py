import csv
import json
from complaint_classifier.train import train


def test_training_pipeline(tmp_path):
    csv_path = tmp_path / "complaints.csv"
    rows = []
    for i in range(12):
        rows.append({"text": f"My credit report contains incorrect information number {i}", "label": "credit_reporting"})
        rows.append({"text": f"Collector called repeatedly about a bill number {i}", "label": "collection_practices"})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    metrics = train(csv_path, tmp_path / "output")
    assert metrics["test_samples"] > 0
    assert (tmp_path / "output" / "model.joblib").exists()
    assert json.loads((tmp_path / "output" / "metrics.json").read_text())["accuracy"] >= 0


def test_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("other,label\nhello,a\n")
    try:
        train(path, tmp_path / "out")
    except ValueError as exc:
        assert "Missing columns" in str(exc)
    else:
        raise AssertionError("Expected missing-column error")
