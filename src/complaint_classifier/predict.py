"""Run inference using a trained model (load only models you trust)."""
import argparse
import joblib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    print(model.predict([args.text])[0])


if __name__ == "__main__":
    main()
