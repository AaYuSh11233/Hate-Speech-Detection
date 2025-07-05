import argparse
from hate_speech.data import load_data
from hate_speech.preprocess import preprocess_df, upsample
from hate_speech.model import build_pipeline, save_model
from hate_speech.evaluate import evaluate
from sklearn.model_selection import train_test_split

def main(train_path, test_path, model_path):
    train, test = load_data(train_path, test_path)

    train = preprocess_df(train)
    test = preprocess_df(test)

    train_balanced = upsample(train)

    X = train_balanced["tweet"]
    y = train_balanced["label"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    f1 = evaluate(y_valid, y_pred)
    print(f"Validation F1 score: {f1:.4f}")

    save_model(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hate Speech Detection Model")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", default="model.joblib")
    args = parser.parse_args()
    main(args.train, args.test, args.model)